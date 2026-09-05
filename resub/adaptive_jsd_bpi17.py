# run_bpi17_validation.py

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon

# ============================================================================
# CONFIGURATION FOR BPI 2017 VALIDATION EXPERIMENT
# ============================================================================
LOG_DIR = Path(__file__).parent / "prelogs"
REP_DIR = Path(__file__).parent / "reports"
SRC_LOG = LOG_DIR / "BPI Challenge 2017.pkl"
OUTPUT_FILENAME = "BPI17_Results_JSD_Validation.xlsx"

# Column mapping for BPI 2017
COLUMNS = {
    'case_id_col': 'case:concept:name',
    'activity_col': 'concept:name',
    'timestamp_col': 'time:timestamp'
}
# Dimension for Mini-Cubes, as decided from data exploration
DIMENSIONS = ['case:ApplicationType']

# Experimental Parameters (kept consistent with original study)
CHUNK_SIZE = 1000 
WINDOW_SIZE = 3000 
STRATEGIES = ['Fidelity', 'Strictness', 'Parsimony']
TRIGGER_THRESHOLD = 0.15 
# ============================================================================


class SmartMiniCube:
    def __init__(self, group, log_subset, case_id_col, activity_col, timestamp_col):
        self.group = group
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        
        self.log = log_subset.copy() if log_subset is not None else pd.DataFrame()
        self.history = []
        
        self.params = {
            'Fidelity':  {'dep': 0.5, 'loop': 0.5, 'sig': 0.1},
            'Strictness': {'dep': 0.5, 'loop': 0.5, 'sig': 0.1},
            'Parsimony': {'dep': 0.5, 'loop': 0.5, 'sig': 0.1}
        }
        
        self.activities = []
        self.last_distribution = None
        self.current_drift_mag = 0.0
        self.is_triggered = False

    def update_data_and_check_trigger(self, new_chunk):
        if self.log.empty: self.log = new_chunk.copy()
        else: self.log = pd.concat([self.log, new_chunk], ignore_index=True)
        
        if len(self.log) > WINDOW_SIZE:
            self.log = self.log.iloc[-WINDOW_SIZE:].copy()
        
        prev_activities = set(self.activities)
        self.activities = sorted(self.log[self.activity_col].unique().tolist())
        
        if not self.activities:
            self.is_triggered = False
            self.current_drift_mag = 0.0
            return

        self.direct_follow = self._build_transition_matrix(self.log, self.activities)
        self.dep_df = self._dependency_graph(self.direct_follow, self.activities)
        current_dist = self._get_probability_vector()
            
        if self.last_distribution is None or set(self.activities) != prev_activities:
            self.is_triggered = True
            self.current_drift_mag = 1.0 # Max drift for new activities
            self.last_distribution = current_dist
        else:
            self.current_drift_mag = jensenshannon(current_dist, self.last_distribution, base=2)
            if self.current_drift_mag > TRIGGER_THRESHOLD:
                self.is_triggered = True
                self.last_distribution = current_dist
            else:
                self.is_triggered = False

    def process_strategy(self, strategy, iteration, cases_seen, ground_truth):
        if not hasattr(self, 'activities') or not self.activities:
            self._record_row(iteration, cases_seen, strategy, None, None, None, 0, 0, 0, ground_truth, 'No Data')
            return
        
        if self.is_triggered:
            best_p = self._run_grid_search(strategy)
            self.params[strategy] = best_p
            criterion = f"Optimized (JSD={self.current_drift_mag:.3f})"
        else:
            criterion = f"Skipped (JSD={self.current_drift_mag:.3f})"

        curr = self.params[strategy]
        final_m = self.adapted_HM(curr['dep'], curr['loop'], curr['sig'])
        ff, pp, ss = self.evaluate(final_m)
        
        self._record_row(iteration, cases_seen, strategy, 
                         curr['dep'], curr['loop'], curr['sig'], 
                         ff, pp, ss, ground_truth, criterion)

    def _run_grid_search(self, strategy):
        dep_range = np.linspace(0.1, 0.9, 10)
        loop_range = np.linspace(0.4, 0.8, 3) 
        sig_range = np.linspace(0.05, 0.25, 4) 
        best_composite = (-np.inf, -np.inf, -np.inf)
        best_p = self.params[strategy].copy()
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
        return best_p

    def _get_probability_vector(self):
        if not hasattr(self, 'direct_follow') or self.direct_follow is None: return np.array([])
        df_np = self.direct_follow.to_numpy().flatten()
        total = np.sum(df_np)
        return df_np / total if total > 0 else df_np

    def _record_row(self, iter, cases, strat, d, l, s, f, p, st, truth, crit):
        self.history.append({
            'iteration': iter, 'cases_seen': cases, 'group': self.group, 'strategy': strat,
            'zone': truth['zone'], 'drift': truth['drift'], 'dep_threshold': d,
            'loop_threshold': l, 'significance_threshold': s, 'fitness_score': f,
            'precision_score': p, 'structure_score': st, 'criterion': crit,
            'drift_magnitude': self.current_drift_mag, 'triggered': 1 if self.is_triggered else 0
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
        dt, lt, st = dep_t or 0.5, loop_t or 0.5, sig_t or 0.1
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
                    if self.dep_df.loc[src, tgt] >= dt and (cnt/total if total > 0 else 0) >= st:
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

class SmartCubeManager:
    def __init__(self, full_log, columns, dimensions, num_variants=2): # Set to 2 for BPI17
        self.columns, self.dimensions, self.full_log = columns, dimensions, full_log
        self.cubes = self._create_variant_cubes(full_log, num_variants)

    def _create_variant_cubes(self, log, num_variants):
        cubes = {}
        dim = self.dimensions[0]
        # Get the top N most frequent variants from the entire log
        top_variants = log[dim].value_counts().nlargest(num_variants).index
        print(f"Initializing cubes for top {num_variants} variants: {top_variants.tolist()}")
        for group in top_variants:
            cubes[group] = SmartMiniCube(group, None, **self.columns)
        return cubes

    def run_smart_experiment(self):
        print(f"Running JSD-Gated Adaptive Miner on BPI 2017 ({STRATEGIES})...")
        dim = self.dimensions[0]
        
        for end_idx in tqdm(range(CHUNK_SIZE, len(self.full_log) + 1, CHUNK_SIZE)):
            new_chunk = self.full_log.iloc[end_idx - CHUNK_SIZE : end_idx]
            iteration = end_idx // CHUNK_SIZE
            
            # Since there is no ground truth, we pass a placeholder
            truth = {'zone': 'Real-World', 'drift': 'N/A'}
            
            for group, cube in self.cubes.items():
                group_chunk = new_chunk[new_chunk[dim] == group]
                if not group_chunk.empty:
                    cube.update_data_and_check_trigger(group_chunk)
                
                for strat in STRATEGIES:
                    cube.process_strategy(strat, iteration, end_idx, truth)
        
        all_histories = [pd.DataFrame(c.history) for c in self.cubes.values()]
        final_df = pd.concat(all_histories, ignore_index=True)
        
        report_path = REP_DIR / OUTPUT_FILENAME
        print(f"\nSaving report to '{report_path}'...")
        with pd.ExcelWriter(report_path) as writer:
            for group in final_df['group'].unique():
                res = final_df[final_df['group'] == group]
                res.sort_values(['strategy', 'iteration']).to_excel(writer, sheet_name=str(group)[:31], index=False)
        
        # Refined Compute Savings Calculation
        total_decision_points = len(final_df['iteration'].unique()) * len(final_df['group'].unique())
        triggered_cycles = final_df.loc[final_df['triggered'] == 1, ['iteration', 'group']].drop_duplicates().shape[0]

        print("\n" + "="*40)
        print("JSD MINER PERFORMANCE ON BPI 2017")
        print("="*40)
        print(f"Total Decision Points (Iterations x Variants): {total_decision_points}")
        print(f"Optimization Cycles Triggered: {triggered_cycles}")
        if total_decision_points > 0:
            savings = 100 * (1 - triggered_cycles / total_decision_points)
            print(f"COMPUTE SAVINGS: {savings:.1f}%")
        print("="*40)

if __name__ == "__main__":
    if not REP_DIR.exists(): REP_DIR.mkdir()
    log_path = SRC_LOG
    if not log_path.exists():
        raise FileNotFoundError(f"ERROR: Input file not found at '{log_path}'. Please run the conversion script first.")
    
    print("Loading and filtering BPI 2017 log...")
    event_log_raw = pd.read_pickle(log_path)
    
    # CRITICAL: Filter for 'complete' events only for meaningful process discovery
    event_log = event_log_raw[event_log_raw['lifecycle:transition'].str.lower() == 'complete'].copy()
    
    print(f"Log loaded. Original events: {len(event_log_raw)}, Using {len(event_log)} 'complete' events.")
    
    # Ensure timestamps are sorted
    event_log = event_log.sort_values(by=COLUMNS['timestamp_col']).reset_index(drop=True)
    
    manager = SmartCubeManager(event_log, COLUMNS, DIMENSIONS)
    manager.run_smart_experiment()
    print("BPI 2017 validation experiment complete.")
