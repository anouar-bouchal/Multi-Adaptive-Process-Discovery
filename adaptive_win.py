import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
LOG_DIR = Path(__file__).parent / "data"
REP_DIR = Path(__file__).parent / "reports"
SRC_LOG = LOG_DIR / "L2026011522_payroll.pkl"
OUTPUT_FILENAME = "R2026011802_Adaptive_Windows.xlsx"

COLUMNS = {'case_id_col': 'case_id', 'activity_col': 'activity', 'timestamp_col': 'timestamp'}
DIMENSIONS = ['bulletin_type']
CHUNK_SIZE = 1000 
WINDOWS_TO_TEST = [1000, 2000, 3000, 4000, 5000]
STRATEGIES = ['Fidelity', 'Strictness', 'Parsimony']
# ============================================================================

class SensitivityCube:
    def __init__(self, window_size):
        self.window_size = window_size
        self.log = pd.DataFrame()
        self.activities = []
        # Store parameters per strategy to keep optimization paths isolated
        self.params = {s: {'dep': 0.5, 'loop': 0.5, 'sig': 0.1} for s in STRATEGIES}

    def update_data(self, new_chunk):
        """Updates log window and rebuilds matrices."""
        if self.log.empty: self.log = new_chunk.copy()
        else: self.log = pd.concat([self.log, new_chunk], ignore_index=True)
        
        # DYNAMIC WINDOW TRIMMING
        if len(self.log) > self.window_size:
            self.log = self.log.iloc[-self.window_size:].copy()
        
        self.activities = sorted(self.log['activity'].unique().tolist())
        if self.activities:
            self.direct_follow = self._build_transition_matrix(self.log, self.activities)
            self.dep_df = self._dependency_graph(self.direct_follow, self.activities)

    def optimize(self, strategy):
        """Runs the specific strategy optimization on the current window data."""
        if not hasattr(self, 'activities') or not self.activities:
            return None

        curr = self.params[strategy]
        
        # Standard Grid Search
        dep_range = np.linspace(0.1, 0.9, 10)
        loop_range = np.linspace(0.4, 0.8, 3) 
        sig_range = np.linspace(0.05, 0.25, 4) 
        
        best_composite = (-np.inf, -np.inf, -np.inf)
        best_p = curr.copy()

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
        
        self.params[strategy] = best_p
        
        # Final Metrics
        final_m = self.adapted_HM(best_p['dep'], best_p['loop'], best_p['sig'])
        ff, pp, ss = self.evaluate(final_m)
        
        return {
            'dep_threshold': best_p['dep'],
            'loop_threshold': best_p['loop'],
            'sig_threshold': best_p['sig'],
            'fitness': ff,
            'precision': pp,
            'structure': ss
        }

    # --- Helpers ---
    def _build_transition_matrix(self, log_df, activities):
        log_sorted = log_df.sort_values(by='timestamp')
        traces = [list(g['activity'].values) for _, g in log_sorted.groupby('case_id')]
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
    
    def adapted_HM(self, dep_t, loop_t, sig_t):
        model = defaultdict(dict)
        outgoing_totals = self.direct_follow.sum(axis=1)
        for src in self.activities:
            total = outgoing_totals.get(src, 0)
            for tgt in self.activities:
                cnt = self.direct_follow.loc[src, tgt] if src in self.direct_follow.index and tgt in self.direct_follow.columns else 0
                if cnt == 0: continue
                if src == tgt:
                    if cnt/(cnt+1) >= loop_t: model[src][tgt] = round(cnt/(cnt+1), 2)
                else:
                    if self.dep_df.loc[src, tgt] >= dep_t and (cnt/total if total>0 else 0) >= sig_t:
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

def run_experiment():
    if not REP_DIR.exists(): REP_DIR.mkdir()
    log_path = SRC_LOG
    if not log_path.exists(): raise FileNotFoundError("Run Generator First")
    
    print("Loading Full Log...")
    full_log = pd.read_pickle(log_path)
    
    dim = DIMENSIONS[0]
    groups = full_log[dim].unique()
    
    # Initialize 5 cubes for EACH group (Key: (group, window_size) -> Cube)
    # This keeps memory clean
    cubes = {}
    for g in groups:
        for w in WINDOWS_TO_TEST:
            cubes[(g, w)] = SensitivityCube(w)
            
    full_history = []

    print("Running Wide Sensitivity Analysis...")
    
    for end_idx in tqdm(range(CHUNK_SIZE, len(full_log) + 1, CHUNK_SIZE)):
        new_chunk = full_log.iloc[end_idx-CHUNK_SIZE : end_idx]
        iteration = end_idx // CHUNK_SIZE
        
        # Ground Truth
        last_event = full_log.iloc[end_idx - 1]
        truth = {
            'zone': last_event.get('zone_label', 'Unknown'),
            'drift': last_event.get('drift_type', 'Unknown')
        }

        # Process each group
        for g in groups:
            group_chunk = new_chunk[new_chunk[dim] == g]
            
            # 1. Update Data for all windows first
            if not group_chunk.empty:
                for w in WINDOWS_TO_TEST:
                    cubes[(g, w)].update_data(group_chunk)
            
            # 2. Run Optimization for each Strategy
            for strat in STRATEGIES:
                # Create one row per (Iteration, Group, Strategy)
                row = {
                    'iteration': iteration,
                    'cases_seen': end_idx,
                    'group': g,
                    'strategy': strat,
                    'zone': truth['zone'],
                    'drift': truth['drift']
                }
                
                # 3. Fill Row with columns from each Window Size
                for w in WINDOWS_TO_TEST:
                    res = cubes[(g, w)].optimize(strat)
                    
                    if res:
                        # Add columns: e.g., dep_threshold_W1000, fitness_score_W1000
                        for k, v in res.items():
                            row[f"{k}_W{w}"] = v
                    else:
                        # Empty placeholders
                        for k in ['dep_threshold','loop_threshold','sig_threshold',
                                  'fitness_score','precision_score','structure_score']:
                            row[f"{k}_W{w}"] = None
                
                full_history.append(row)

    print("\nCompiling Report...")
    final_df = pd.DataFrame(full_history)
    
    # Organize columns nicely
    base_cols = ['iteration', 'cases_seen', 'group', 'strategy', 'zone', 'drift']
    
    # Sort remaining columns by Metric first, then Window
    # e.g., fitness_score_W1000, fitness_score_W2000... precision_score_W1000...
    metric_cols = [c for c in final_df.columns if c not in base_cols]
    
    def sort_key(col_name):
        parts = col_name.split('_W')
        if len(parts) < 2: return (99, col_name)
        base = parts[0]
        win = int(parts[1])
        # Group parameters together, then scores
        order = 0 if 'threshold' in base else 1
        return (order, base, win)
        
    metric_cols.sort(key=sort_key)
    
    final_df = final_df[base_cols + metric_cols]
    
    final_df.to_excel(REP_DIR / OUTPUT_FILENAME, index=False)
    print(f"Saved Wide Report to: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    run_experiment()