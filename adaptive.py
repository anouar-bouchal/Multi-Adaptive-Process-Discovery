import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

LOG_DIR = Path(__file__).parent / "data"
REP_DIR = Path(__file__).parent / "reports"
SRC_LOG = LOG_DIR / "L2026011522_payroll.pkl"
OUTPUT_FILENAME = REP_DIR / "R2026011802_Adaptive.xlsx"

COLUMNS = {'case_id_col': 'case_id', 'activity_col': 'activity', 'timestamp_col': 'timestamp'}
DIMENSIONS = ['bulletin_type']
CHUNK_SIZE = 1000 
WINDOW_SIZE = 3000 

class MiniCube:
    def __init__(self, group, log_subset, case_id_col, activity_col, timestamp_col):
        self.group = group
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.log = log_subset.copy() if log_subset is not None else pd.DataFrame()
        self.history = []
        self.current_dep, self.current_loop, self.current_sig = 0.5, 0.5, 0.1
        self.activities = []

    def reset(self):
        self.log = pd.DataFrame()
        self.history = []
        self.current_dep, self.current_loop, self.current_sig = 0.5, 0.5, 0.1
        self.activities = []

    def update_data(self, new_chunk):
        if self.log.empty: self.log = new_chunk.copy()
        else: self.log = pd.concat([self.log, new_chunk], ignore_index=True)
        
        if len(self.log) > WINDOW_SIZE:
            self.log = self.log.iloc[-WINDOW_SIZE:].copy()
        
        self.activities = sorted(self.log[self.activity_col].unique().tolist())
        if self.activities:
            self.direct_follow = self._build_transition_matrix(self.log, self.activities)
            self.dep_df = self._dependency_graph(self.direct_follow, self.activities)

    def optimize_and_record(self, strategy, iteration, cases_seen, ground_truth):
        # ground_truth is a dict with {zone, drift_type}
        
        if not hasattr(self, 'activities') or not self.activities:
            self._record_row(iteration, cases_seen, strategy, None, None, None, 0, 0, 0, ground_truth, 'No Data')
            return

        dep_range = np.linspace(0.1, 0.9, 10)
        loop_range = np.linspace(0.4, 0.8, 3) 
        sig_range = np.linspace(0.05, 0.25, 4) 
        
        best_composite = (-np.inf, -np.inf, -np.inf)
        best_p = {'dep': self.current_dep, 'loop': self.current_loop, 'sig': self.current_sig}

        for d in dep_range:
            for l in loop_range:
                for s in sig_range:
                    m = self.adapted_HM(d, l, s)
                    f, p, st = self.evaluate(m)
                    
                    if strategy == 'Fidelity': cur = (f, p, st)
                    elif strategy == 'Strictness': cur = (p, f, st)
                    else: cur = (st, f, p)
                    
                    if cur > best_composite:
                        best_composite = cur
                        best_p = {'dep': d, 'loop': l, 'sig': s}
        
        self.current_dep, self.current_loop, self.current_sig = best_p['dep'], best_p['loop'], best_p['sig']
        final_m = self.adapted_HM()
        ff, pp, ss = self.evaluate(final_m)
        
        self._record_row(iteration, cases_seen, strategy, 
                         self.current_dep, self.current_loop, self.current_sig, 
                         ff, pp, ss, ground_truth, f"Windowed ({WINDOW_SIZE})")

    def _record_row(self, iter, cases, strat, d, l, s, f, p, st, truth, crit):
        self.history.append({
            'iteration': iter,
            'cases_seen': cases,
            'group': self.group,
            'strategy': strat,
            'zone': truth['zone'],
            'drift': truth['drift'],
            'dep_threshold': d, 'loop_threshold': l, 'sig_threshold': s,
            'adaptive fitness': f, 'adaptive precision': p, 'adaptive structure': st,
            'criterion': crit
        })

    def _build_transition_matrix(self, log_df, activities):
        log_sorted = log_df.sort_values(by=self.timestamp_col)
        traces = [list(g[self.activity_col].values) for _, g in log_sorted.groupby(self.case_id_col)]
        act_index = {act: i for i, act in enumerate(activities)}
        df_matrix = np.zeros((len(activities), len(activities)), dtype=int)
        for trace in traces:
            for i in range(len(trace) - 1):
                if trace[i] in act_index and trace[i+1] in act_index:
                    df_matrix[act_index[trace[i]], act_index[trace[i+1]]] += 1
        return pd.DataFrame(df_matrix, index=activities, columns=activities)

    def _dependency_graph(self, direct_follow_df, activities):
        fc = direct_follow_df.reindex(index=activities, columns=activities, fill_value=0)
        dep_graph = []
        for act_i in activities:
            dep_row = []
            for act_j in activities:
                a_ij, a_ji = fc.loc[act_i, act_j], fc.loc[act_j, act_i]
                dep_row.append(round((a_ij - a_ji) / (a_ij + a_ji + 1), 2))
            dep_graph.append(dep_row)
        return pd.DataFrame(dep_graph, index=activities, columns=activities)
    
    def adapted_HM(self, dep_t=None, loop_t=None, sig_t=None):
        dt = dep_t if dep_t is not None else self.current_dep
        lt = loop_t if loop_t is not None else self.current_loop
        st = sig_t if sig_t is not None else self.current_sig
        model = defaultdict(dict)
        outgoing_totals = self.direct_follow.sum(axis=1)
        for src in self.activities:
            total = outgoing_totals.get(src, 0)
            for tgt in self.activities:
                cnt = self.direct_follow.loc[src, tgt] if src in self.direct_follow.index and tgt in self.direct_follow.columns else 0
                if cnt == 0: continue
                if src == tgt:
                    if cnt/(cnt+1) >= lt: model[src][tgt] = round(cnt/(cnt+1), 2)
                else:
                    if self.dep_df.loc[src, tgt] >= dt and (cnt/total if total>0 else 0) >= st:
                        model[src][tgt] = round(self.dep_df.loc[src, tgt], 2)
        return dict(model)
    
    def evaluate(self, model):
        return self._fitness(model, self.direct_follow), self._precision(model, self.dep_df), self._structure_score(model, self.direct_follow)

    def _fitness(self, model, direct_follow):
        total_rel = (direct_follow > 0).to_numpy().sum()
        if total_rel == 0: return 1.0
        valid = 0
        for src in direct_follow.index:
            for tgt in direct_follow.columns:
                if direct_follow.loc[src, tgt] > 0 and src in model and tgt in model.get(src, {}):
                    valid += 1
        return valid / total_rel

    def _precision(self, model, dep_df):
        if not model: return 1.0
        edges, total_dep = 0, 0.0
        for src, targets in model.items():
            for tgt in targets:
                edges += 1
                if src in dep_df.index and tgt in dep_df.columns and dep_df.loc[src, tgt] > 0:
                    total_dep += dep_df.loc[src, tgt]
        return total_dep / edges if edges > 0 else 1.0

    def _structure_score(self, model, direct_follow):
        splits, total_qual = 0, 0.0
        for act, edges in model.items():
            succ = list(edges.keys())
            if len(succ) > 1:
                splits += 1
                qual, pairs = 0.0, 0
                for i in range(len(succ)):
                    for j in range(i+1, len(succ)):
                        b, c = succ[i], succ[j]
                        b_c = direct_follow.loc[b, c] if b in direct_follow.index and c in direct_follow.columns else 0
                        c_b = direct_follow.loc[c, b] if c in direct_follow.index and b in direct_follow.columns else 0
                        a_b, a_c = direct_follow.loc[act, b], direct_follow.loc[act, c]
                        qual += (1 - ((b_c + c_b)/(a_b + a_c + 1)))
                        pairs += 1
                if pairs > 0: total_qual += max(0, qual/pairs)
        return total_qual / splits if splits > 0 else 1.0

class MiniCubeManager:
    def __init__(self, full_log, columns, dimensions, num_variants=3):
        self.columns, self.dimensions, self.full_log = columns, dimensions, full_log
        self.cubes = self._create_variant_cubes(full_log, num_variants)
        self.full_history = []

    def _create_variant_cubes(self, log, num_variants):
        cubes = {}
        dim = self.dimensions[0]
        for group in log[dim].value_counts().nlargest(num_variants).index:
            cubes[group] = MiniCube(group, None, **self.columns)
        return cubes

    def run_full_experiment(self, strategies=['Fidelity', 'Strictness', 'Parsimony']):
        for strategy in strategies:
            print(f"\n--- Strategy: {strategy} ---")
            for cube in self.cubes.values(): cube.reset()
            
            dim = self.dimensions[0]
            for end_idx in tqdm(range(CHUNK_SIZE, len(self.full_log) + 1, CHUNK_SIZE)):
                new_chunk = self.full_log.iloc[end_idx-CHUNK_SIZE : end_idx]
                iteration = end_idx // CHUNK_SIZE
                
                # EXTRACT GROUND TRUTH from the LAST event in this chunk
                # This tells us "Where are we right now?"
                last_event = self.full_log.iloc[end_idx - 1]
                truth = {
                    'zone': last_event.get('zone_label', 'Unknown'),
                    'drift': last_event.get('drift_type', 'Unknown')
                }
                
                for group, cube in self.cubes.items():
                    group_chunk = new_chunk[new_chunk[dim] == group]
                    if not group_chunk.empty:
                        cube.update_data(group_chunk)
                    
                    cube.optimize_and_record(strategy, iteration, end_idx, truth)
            
            for cube in self.cubes.values():
                self.full_history.append(pd.DataFrame(cube.history))
        return pd.concat(self.full_history, ignore_index=True) if self.full_history else pd.DataFrame()

if __name__ == "__main__":
    event_log = pd.read_pickle(SRC_LOG)
    manager = MiniCubeManager(event_log, COLUMNS, DIMENSIONS)
    res = manager.run_full_experiment(['Fidelity', 'Strictness', 'Parsimony'])
    
    with pd.ExcelWriter(OUTPUT_FILENAME) as writer:
        for group in res['group'].unique():
            res[res['group'] == group].sort_values(['strategy', 'iteration']).to_excel(writer, sheet_name=str(group)[:31], index=False)
    print("Done.")