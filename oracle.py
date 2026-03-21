import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

LOG_DIR = Path(__file__).parent / "data"
REP_DIR = Path(__file__).parent / "reports"
SRC_LOG = LOG_DIR / "L2026011522_payroll.pkl"
OUTPUT_FILENAME = REP_DIR / "R2026011802_Oracle.xlsx"

COLUMNS = {'case_id_col': 'case_id', 'activity_col': 'activity', 'timestamp_col': 'timestamp'}
DIMENSIONS = ['bulletin_type']
CHUNK_SIZE = 1000 
WINDOW_SIZE = 3000  # Matched to Adaptive

class StaticMiniCube:
    def __init__(self, group, full_group_log, case_id_col, activity_col, timestamp_col):
        self.group = group
        self.full_log = full_group_log.copy()
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        
        # Training State
        self.optimal_hyperparameters = {}
        
        # Evaluation State (Windowed)
        self.log = pd.DataFrame()
        self.activities = []
        self.history = []

    # --- PHASE 1: TRAIN ON FULL LOG (The "Time Traveler" Aspect) ---
    def find_optimal_hyperparameters_on_full_log(self, strategies=['Fidelity', 'Strictness', 'Parsimony']):
        print(f"  Finding optimal hyperparameters for '{self.group}' (FULL LOG)...")
        activities = sorted(self.full_log[self.activity_col].unique().tolist())
        if not activities: return
        
        # Build matrices on the FULL dataset once
        direct_follow = self._build_transition_matrix(self.full_log, activities)
        dep_df = self._dependency_graph(direct_follow, activities)
        
        dep_thresholds = np.linspace(0.1, 0.9, 10)
        loop_thresholds = np.linspace(0.4, 0.8, 3)
        sig_thresholds = np.linspace(0.05, 0.25, 4)
        
        for strategy in strategies:
            best_log_score = -np.inf
            best_params = {'dep': 0.5, 'loop': 0.5, 'sig': 0.1}
            
            # Grid Search
            for dep_t in dep_thresholds:
                for loop_t in loop_thresholds:
                    for sig_t in sig_thresholds:
                        model = self.adapted_HM(direct_follow, dep_df, activities, dep_t, loop_t, sig_t)
                        f, p, s = self.evaluate(model, direct_follow, dep_df)
                        
                        e = 1e-9
                        if strategy == 'Fidelity': score = np.log(f + e)
                        elif strategy == 'Strictness': score = np.log(p + e)
                        else: score = np.log(s + e)
                        
                        if score > best_log_score:
                            best_log_score = score
                            best_params = {'dep': dep_t, 'loop': loop_t, 'sig': sig_t}
            
            self.optimal_hyperparameters[strategy] = best_params

    # --- PHASE 2: EVALUATION (Exact same logic as Adaptive) ---
    def update_data(self, new_chunk):
        """Maintain the sliding window exactly like Adaptive miner."""
        if self.log.empty: 
            self.log = new_chunk.copy()
        else: 
            self.log = pd.concat([self.log, new_chunk], ignore_index=True)
        
        if len(self.log) > WINDOW_SIZE:
            self.log = self.log.iloc[-WINDOW_SIZE:].copy()
        
        self.activities = sorted(self.log[self.activity_col].unique().tolist())
        if self.activities:
            self.direct_follow = self._build_transition_matrix(self.log, self.activities)
            self.dep_df = self._dependency_graph(self.direct_follow, self.activities)

    def evaluate_with_frozen_params(self, strategy, iteration, cases_seen, ground_truth):
        """
        Evaluate the CURRENT WINDOW using the FROZEN GLOBAL PARAMETERS.
        This compares how well the 'Global Truth' fits the 'Local Reality'.
        """
        if not hasattr(self, 'activities') or not self.activities:
            self._record_row(iteration, cases_seen, strategy, None, None, None, 0, 0, 0, ground_truth, 'No Data')
            return
        
        if strategy not in self.optimal_hyperparameters:
            return

        # Retrieve Frozen Params
        params = self.optimal_hyperparameters[strategy]
        
        # Build Model using Window Matrices (Local Data) + Frozen Params (Global Knowledge)
        model = self.adapted_HM(self.direct_follow, self.dep_df, self.activities, 
                                params['dep'], params['loop'], params['sig'])
        
        # Evaluate
        f, p, s = self.evaluate(model, self.direct_follow, self.dep_df)
        
        self._record_row(iteration, cases_seen, strategy, 
                         params['dep'], params['loop'], params['sig'], 
                         f, p, s, ground_truth, f"Static Frozen (Win {WINDOW_SIZE})")

    def _record_row(self, iter, cases, strat, d, l, s, f, p, st, truth, crit):
        self.history.append({
            'iteration': iter,
            'cases_seen': cases,
            'group': self.group,
            'strategy': strat,
            'zone': truth['zone'],
            'drift': truth['drift'],
            'dep_threshold': d, 'loop_threshold': l, 'sig_threshold': s,
            'oracle fitness': f, 'oracle precision': p, 'oracle structure': st,
            'criterion': crit
        })

    # --- Matrix & Mining Logic (Identical to Adaptive) ---
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

    def adapted_HM(self, direct_follow, dep_df, activities, dep_t, loop_t, sig_t):
        model = defaultdict(dict)
        outgoing_totals = direct_follow.sum(axis=1)
        for src in activities:
            total = outgoing_totals.get(src, 0)
            for tgt in activities:
                cnt = direct_follow.loc[src, tgt] if src in direct_follow.index and tgt in direct_follow.columns else 0
                if cnt == 0: continue
                if src == tgt:
                    if cnt/(cnt+1) >= loop_t: model[src][tgt] = round(cnt/(cnt+1), 2)
                else:
                    if dep_df.loc[src, tgt] >= dep_t and (cnt/total if total>0 else 0) >= sig_t:
                        model[src][tgt] = round(dep_df.loc[src, tgt], 2)
        return dict(model)

    def evaluate(self, model, direct_follow, dep_df):
        return self._fitness(model, direct_follow), self._precision(model, dep_df), self._structure_score(model, direct_follow)
    
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

class StaticMiniCubeManager:
    def __init__(self, full_log, columns, dimensions, num_variants=3):
        self.columns, self.dimensions, self.full_log = columns, dimensions, full_log
        self.cubes = self._create_static_cubes(full_log, num_variants)

    def _create_static_cubes(self, log, num_variants):
        cubes = {}
        dim = self.dimensions[0]
        for group in log[dim].value_counts().nlargest(num_variants).index:
            cubes[group] = StaticMiniCube(group, log[log[dim] == group], **self.columns)
        return cubes

    def run_static_experiment(self, strategies=['Fidelity', 'Strictness', 'Parsimony']):
        print("Training Static Time-Traveler (Optimizing on FULL Log)...")
        # 1. Train on Full Log (The Static Part)
        for group, cube in self.cubes.items():
            cube.find_optimal_hyperparameters_on_full_log(strategies)
        
        print("Evaluating on Sliding Window (The Reality Check)...")
        dim = self.dimensions[0]
        
        # 2. Iterate through log chunks (Evaluation Part)
        for end_idx in tqdm(range(CHUNK_SIZE, len(self.full_log) + 1, CHUNK_SIZE)):
            new_chunk = self.full_log.iloc[end_idx-CHUNK_SIZE : end_idx]
            iteration = end_idx // CHUNK_SIZE
            
            # GET TRUTH
            last_event = self.full_log.iloc[end_idx - 1]
            truth = {
                'zone': last_event.get('zone_label', 'Unknown'),
                'drift': last_event.get('drift_type', 'Unknown')
            }
            
            for group, cube in self.cubes.items():
                group_chunk = new_chunk[new_chunk[dim] == group]
                if not group_chunk.empty:
                    cube.update_data(group_chunk)
                
                # Evaluate for each strategy using the FROZEN parameters
                for strategy in strategies:
                    cube.evaluate_with_frozen_params(strategy, iteration, end_idx, truth)
        
        all_h = [pd.DataFrame(c.history) for c in self.cubes.values() if c.history]
        return pd.concat(all_h, ignore_index=True) if all_h else pd.DataFrame()

if __name__ == "__main__":
    event_log = pd.read_pickle(SRC_LOG)
    manager = StaticMiniCubeManager(event_log, COLUMNS, DIMENSIONS)
    res = manager.run_static_experiment()
    
    with pd.ExcelWriter(OUTPUT_FILENAME) as writer:
        for group in res['group'].unique():
            res[res['group'] == group].sort_values(['strategy', 'iteration']).to_excel(writer, sheet_name=str(group)[:31], index=False)
    print("Done.")