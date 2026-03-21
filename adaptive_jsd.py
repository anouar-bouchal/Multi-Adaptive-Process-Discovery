import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon

# ============================================================================
# CONFIGURATION
# ============================================================================
LOG_DIR = Path(__file__).parent / "data"
REP_DIR = Path(__file__).parent / "reports"
SRC_LOG = LOG_DIR / "L2026011522_payroll.pkl"
OUTPUT_FILENAME = "R2026011802_Adaptive_JSD.xlsx"

COLUMNS = {'case_id_col': 'case_id', 'activity_col': 'activity', 'timestamp_col': 'timestamp'}
DIMENSIONS = ['bulletin_type']
CHUNK_SIZE = 1000 
WINDOW_SIZE = 3000 
STRATEGIES = ['Fidelity', 'Strictness', 'Parsimony']

# SOTA DETECTOR SETTINGS
# Jensen-Shannon is bounded [0, 1]. A score > 0.15 usually implies structural drift.
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
        
        # STATE: Separate parameters for EACH strategy
        # This ensures Precision strategy doesn't overwrite Fidelity parameters
        self.params = {
            'Fidelity':  {'dep': 0.5, 'loop': 0.5, 'sig': 0.1},
            'Strictness': {'dep': 0.5, 'loop': 0.5, 'sig': 0.1},
            'Parsimony': {'dep': 0.5, 'loop': 0.5, 'sig': 0.1}
        }
        
        self.activities = []
        
        # SOTA MEMORY
        self.last_distribution = None
        self.current_drift_mag = 0.0
        self.is_triggered = False

    def update_data_and_check_trigger(self, new_chunk):
        """
        1. Updates Window
        2. Rebuilds Matrices
        3. Calculates Jensen-Shannon Divergence
        4. Sets self.is_triggered flag
        """
        # --- Update Window ---
        if self.log.empty: self.log = new_chunk.copy()
        else: self.log = pd.concat([self.log, new_chunk], ignore_index=True)
        
        if len(self.log) > WINDOW_SIZE:
            self.log = self.log.iloc[-WINDOW_SIZE:].copy()
        
        self.activities = sorted(self.log[self.activity_col].unique().tolist())
        
        # --- Build Matrices ---
        if self.activities:
            self.direct_follow = self._build_transition_matrix(self.log, self.activities)
            self.dep_df = self._dependency_graph(self.direct_follow, self.activities)
            
            # --- SOTA Detection (JSD) ---
            current_dist = self._get_probability_vector()
            
            if self.last_distribution is None:
                self.is_triggered = True
                self.current_drift_mag = 1.0
                self.last_distribution = current_dist
            else:
                # Handle dimension mismatch (e.g., new activity appeared)
                if current_dist.shape != self.last_distribution.shape:
                    self.is_triggered = True
                    self.current_drift_mag = 1.0
                    self.last_distribution = current_dist
                else:
                    # Jensen-Shannon Divergence (Base 2)
                    self.current_drift_mag = jensenshannon(current_dist, self.last_distribution, base=2)
                    
                    if self.current_drift_mag > TRIGGER_THRESHOLD:
                        self.is_triggered = True
                        self.last_distribution = current_dist # Update Baseline
                    else:
                        self.is_triggered = False
        else:
            self.is_triggered = False
            self.current_drift_mag = 0.0

    def process_strategy(self, strategy, iteration, cases_seen, ground_truth):
        """
        Runs the optimization logic for a SPECIFIC strategy.
        Uses the shared trigger flag.
        """
        if not hasattr(self, 'activities') or not self.activities:
            self._record_row(iteration, cases_seen, strategy, None, None, None, 0, 0, 0, ground_truth, 'No Data')
            return

        # LOGIC:
        # If Triggered: Run Grid Search for THIS strategy -> Update THIS strategy's params
        # If Not: Keep THIS strategy's old params
        
        if self.is_triggered:
            best_p = self._run_grid_search(strategy)
            self.params[strategy] = best_p
            criterion = f"Optimized (JSD={self.current_drift_mag:.3f})"
        else:
            criterion = f"Skipped (JSD={self.current_drift_mag:.3f})"

        # ALWAYS Evaluate (to prove the skipped parameters still work)
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
        # Start search from current parameters to ensure stability
        best_p = self.params[strategy].copy()

        for d in dep_range:
            for l in loop_range:
                for s in sig_range:
                    m = self.adapted_HM(d, l, s)
                    f, p, st = self.evaluate(m)
                    
                    # Tuple Optimization (Tie-Breaking)
                    if strategy == 'Fidelity': cur = (f, p, st)
                    elif strategy == 'Strictness': cur = (p, f, st)
                    else: cur = (st, f, p)
                    
                    if cur > best_composite:
                        best_composite = cur
                        best_p = {'dep': d, 'loop': l, 'sig': s}
        return best_p

    def _get_probability_vector(self):
        """Helper for JSD."""
        if not hasattr(self, 'direct_follow') or self.direct_follow is None: return None
        df_np = self.direct_follow.to_numpy().flatten()
        total = np.sum(df_np)
        if total == 0: return df_np
        return df_np / total

    def _record_row(self, iter, cases, strat, d, l, s, f, p, st, truth, crit):
        # EXACT STANDARD FORMAT + Smart Columns
        self.history.append({
            'iteration': iter,
            'cases_seen': cases,
            'group': self.group,
            'strategy': strat,
            'zone': truth['zone'],
            'drift': truth['drift'],
            'dep_threshold': d,
            'loop_threshold': l,
            'significance_threshold': s,
            'fitness_score': f,
            'precision_score': p,
            'structure_score': st,
            'criterion': crit,
            # Appended Columns
            'drift_magnitude': self.current_drift_mag,
            'triggered': 1 if self.is_triggered else 0
        })

    # --- Standard Helpers ---
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
        dt = dep_t if dep_t is not None else 0.5
        lt = loop_t if loop_t is not None else 0.5
        st = sig_t if sig_t is not None else 0.1
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

class SmartCubeManager:
    def __init__(self, full_log, columns, dimensions, num_variants=3):
        self.columns, self.dimensions, self.full_log = columns, dimensions, full_log
        self.cubes = self._create_variant_cubes(full_log, num_variants)

    def _create_variant_cubes(self, log, num_variants):
        cubes = {}
        dim = self.dimensions[0]
        # Create cubes for top variants found in the WHOLE log
        for group in log[dim].value_counts().nlargest(num_variants).index:
            cubes[group] = SmartMiniCube(group, None, **self.columns)
        return cubes

    def run_smart_experiment(self):
        print(f"Running SOTA Smart Adaptive ({STRATEGIES})...")
        dim = self.dimensions[0]
        
        for end_idx in tqdm(range(CHUNK_SIZE, len(self.full_log) + 1, CHUNK_SIZE)):
            new_chunk = self.full_log.iloc[end_idx-CHUNK_SIZE : end_idx]
            iteration = end_idx // CHUNK_SIZE
            last_event = self.full_log.iloc[end_idx - 1]
            truth = {'zone': last_event.get('zone_label', 'Unknown'), 'drift': last_event.get('drift_type', 'Unknown')}
            
            for group, cube in self.cubes.items():
                group_chunk = new_chunk[new_chunk[dim] == group]
                
                # 1. Update Data & Trigger SOTA Detector (Shared)
                if not group_chunk.empty:
                    cube.update_data_and_check_trigger(group_chunk)
                
                # 2. Run All Strategies
                for strat in STRATEGIES:
                    cube.process_strategy(strat, iteration, end_idx, truth)
        
        # Collect
        all_histories = []
        for cube in self.cubes.values():
            all_histories.append(pd.DataFrame(cube.history))
        
        final_df = pd.concat(all_histories, ignore_index=True)
        
        # Save Report
        with pd.ExcelWriter(REP_DIR / OUTPUT_FILENAME) as writer:
            for group in final_df['group'].unique():
                res = final_df[final_df['group'] == group]
                res.sort_values(['strategy', 'iteration']).to_excel(writer, sheet_name=str(group)[:31], index=False)
        
        # ROI Stats
        total_rows = len(final_df)
        triggered_rows = final_df['triggered'].sum()
        print("\n" + "="*40)
        print(f"SMART MINER (Jensen-Shannon) PERFORMANCE")
        print("="*40)
        print(f"Total Iterations: {total_rows}")
        print(f"Triggered: {triggered_rows}")
        print(f"Skipped: {total_rows - triggered_rows}")
        print(f"COMPUTE SAVINGS: {100 * (1 - triggered_rows/total_rows):.1f}%")
        print("="*40)

if __name__ == "__main__":
    if not REP_DIR.exists(): REP_DIR.mkdir()
    log_path = SRC_LOG
    if not log_path.exists(): raise FileNotFoundError("Run Generator First")
    
    event_log = pd.read_pickle(log_path)
    manager = SmartCubeManager(event_log, COLUMNS, DIMENSIONS)
    manager.run_smart_experiment()
    print("Done.")