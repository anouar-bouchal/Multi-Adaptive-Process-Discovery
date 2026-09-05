import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon

# ============================================================================
# CONFIGURATION - DETERMINISTIC BASELINE (GRID SEARCH)
# ============================================================================
LOG_DIR = Path(__file__).parent / "prelogs"
REP_DIR = Path(__file__).parent / "reports"
SRC_LOG = LOG_DIR / "BPI Challenge 2017.pkl"
OUTPUT_FILENAME = "BPI17_Results_GRID_AUDIT.xlsx"

COLUMNS = {
    'case_id_col': 'case:concept:name',
    'activity_col': 'concept:name',
    'timestamp_col': 'time:timestamp'
}
DIMENSIONS = ['case:ApplicationType']
STRATEGIES = ['Fidelity', 'Strictness', 'Parsimony']

# Experimental Constants
CHUNK_SIZE = 1000 
WINDOW_SIZE = 3000 
TRIGGER_THRESHOLD = 0.15 

# SEARCH SPACE DEFINITION (Exhaustive)
# 9 (dep) * 3 (loop) * 5 (sig) = 135 Evaluations
DEP_RANGE = np.linspace(0.1, 0.9, 9)
LOOP_RANGE = np.linspace(0.4, 0.8, 3) 
SIG_RANGE = np.linspace(0.05, 0.25, 5) 
GRID_EVAL_BUDGET = len(DEP_RANGE) * len(LOOP_RANGE) * len(SIG_RANGE)

# ============================================================================

class SmartMiniCube:
    def __init__(self, group, case_id_col, activity_col, timestamp_col):
        self.group = group
        self.case_id_col, self.activity_col, self.timestamp_col = case_id_col, activity_col, timestamp_col
        self.log, self.history, self.activities = pd.DataFrame(), [], []
        self.params = {strat: {'dep': 0.5, 'loop': 0.5, 'sig': 0.1} for strat in STRATEGIES}
        self.last_distribution, self.current_drift_mag, self.is_triggered = None, 0.0, False

    def update_data_and_check_trigger(self, new_chunk):
        """Standard JSD-gating logic (unaltered)."""
        if self.log.empty: self.log = new_chunk.copy()
        else: self.log = pd.concat([self.log, new_chunk], ignore_index=True)
        if len(self.log) > WINDOW_SIZE: self.log = self.log.iloc[-WINDOW_SIZE:].copy()
        
        prev_act = set(self.activities)
        self.activities = sorted(self.log[self.activity_col].unique().tolist())
        if not self.activities: return

        self.direct_follow = self._build_transition_matrix(self.log, self.activities)
        self.dep_df = self._dependency_graph(self.direct_follow, self.activities)
        curr_dist = self._get_probability_vector()
            
        if self.last_distribution is None or set(self.activities) != prev_act:
            self.is_triggered, self.current_drift_mag = True, 1.0 
        else:
            self.current_drift_mag = jensenshannon(curr_dist, self.last_distribution, base=2)
            self.is_triggered = self.current_drift_mag > TRIGGER_THRESHOLD
        if self.is_triggered: self.last_distribution = curr_dist

    def process_strategy(self, strategy, iteration):
        """Processes the strategy and records the hardware-independent audit metrics."""
        if not self.activities: return
        
        evals_used = 0
        if self.is_triggered:
            best_p = self._run_grid_search(strategy)
            evals_used = GRID_EVAL_BUDGET # Hardware-independent computational work
            self.params[strategy] = best_p
            crit = "Optimized"
        else:
            crit = "Skipped"

        curr = self.params[strategy]
        final_m = self.adapted_HM(curr['dep'], curr['loop'], curr['sig'])
        f, p, s = self.evaluate(final_m)
        
        # Calculate Audit Metric: Target Score / Computational Work
        target_score = f if strategy == 'Fidelity' else p if strategy == 'Strictness' else s
        efficiency = (target_score / evals_used) if evals_used > 0 else 0

        self.history.append({
            'iteration': iteration, 'group': self.group, 'strategy': strategy,
            'fitness_score': f, 'precision_score': p, 'structure_score': s,
            'dep_threshold': curr['dep'], 'loop_threshold': curr['loop'], 'significance_threshold': curr['sig'],
            'criterion': crit, 'drift_magnitude': self.current_drift_mag,
            # AUDIT OUTPUTS
            'opt_algo': 'Grid Search (Deterministic)',
            'eval_budget': evals_used,
            'efficiency_index': efficiency
        })

    def _run_grid_search(self, strategy):
        best_composite = (-np.inf, -np.inf, -np.inf)
        best_p = self.params[strategy].copy()
        for d in DEP_RANGE:
            for l in LOOP_RANGE:
                for s in SIG_RANGE:
                    m = self.adapted_HM(d, l, s)
                    f, p, st = self.evaluate(m)
                    if strategy == 'Fidelity': cur = (f, p, st)
                    elif strategy == 'Strictness': cur = (p, f, st)
                    else: cur = (st, f, p)
                    if cur > best_composite:
                        best_composite = cur
                        best_p = {'dep': d, 'loop': l, 'sig': s}
        return best_p

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
                dep_row.append((a_ij - a_ji) / (a_ij + a_ji + 1))
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
                    if self.dep_df.loc[src, tgt] >= dep_t and (cnt/total if total > 0 else 0) >= sig_t:
                        model[src][tgt] = round(self.dep_df.loc[src, tgt], 2)
        return dict(model)
    
    def evaluate(self, model):
        return self._fitness(model, self.direct_follow), self._precision(model, self.dep_df), self._structure_score(model, self.direct_follow)

    def _fitness(self, model, df):
        total = (df > 0).to_numpy().sum()
        if total == 0: return 1.0
        valid = sum(1 for src in df.index for tgt in df.columns if df.loc[src, tgt] > 0 and src in model and tgt in model.get(src, {}))
        return valid / total

    def _precision(self, model, dep):
        if not model: return 1.0
        edges, total_dep = 0, 0.0
        for src, targets in model.items():
            for tgt in targets:
                edges += 1
                if src in dep.index and tgt in dep.columns and dep.loc[src, tgt] > 0:
                    total_dep += dep.loc[src, tgt]
        return total_dep / edges if edges > 0 else 1.0

    def _structure_score(self, model, df):
        splits, total_qual = 0, 0.0
        for act, edges in model.items():
            succ = list(edges.keys())
            if len(succ) > 1:
                splits += 1
                qual, pairs = 0.0, 0
                for i in range(len(succ)):
                    for j in range(i+1, len(succ)):
                        b, c = succ[i], succ[j]
                        b_c = df.loc[b, c] if b in df.index and c in df.columns else 0
                        c_b = df.loc[c, b] if c in df.index and b in df.columns else 0
                        a_b, a_c = df.loc[act, b], df.loc[act, c]
                        qual += (1 - ((b_c + c_b)/(a_b + a_c + 1)))
                        pairs += 1
                if pairs > 0: total_qual += max(0, qual/pairs)
        return total_qual / splits if splits > 0 else 1.0

    def _get_probability_vector(self):
        df_np = self.direct_follow.to_numpy().flatten()
        total = np.sum(df_np)
        return df_np / total if total > 0 else df_np

class AuditManager:
    def __init__(self, full_log, columns, dimensions, num_variants=2):
        self.full_log = full_log
        self.cubes = {g: SmartMiniCube(g, **columns) for g in full_log[dimensions[0]].value_counts().nlargest(num_variants).index}

    def run(self):
        dim = DIMENSIONS[0]
        for end_idx in tqdm(range(CHUNK_SIZE, len(self.full_log) + 1, CHUNK_SIZE)):
            chunk = self.full_log.iloc[end_idx - CHUNK_SIZE : end_idx]
            it = end_idx // CHUNK_SIZE
            for group, cube in self.cubes.items():
                g_chunk = chunk[chunk[dim] == group]
                if not g_chunk.empty: cube.update_data_and_check_trigger(g_chunk)
                for strat in STRATEGIES: cube.process_strategy(strat, it)
        
        all_res = pd.concat([pd.DataFrame(c.history) for c in self.cubes.values()], ignore_index=True)
        with pd.ExcelWriter(REP_DIR / OUTPUT_FILENAME) as writer:
            for group in all_res['group'].unique():
                all_res[all_res['group'] == group].to_excel(writer, sheet_name=str(group)[:31], index=False)

if __name__ == "__main__":
    if not REP_DIR.exists(): REP_DIR.mkdir()
    event_log = pd.read_pickle(SRC_LOG)
    event_log = event_log[event_log['lifecycle:transition'].str.lower() == 'complete'].copy()
    event_log = event_log.sort_values(by=COLUMNS['timestamp_col']).reset_index(drop=True)
    AuditManager(event_log, COLUMNS, DIMENSIONS).run()
    print("Grid Search Audit Complete.")
