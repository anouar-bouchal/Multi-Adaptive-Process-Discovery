import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION: REALISTIC NON-LINEAR BASELINE
# ============================================================================
OUTPUT_FILENAME = "L2026011603_payroll"
NUM_CASES = 50000
START_DATE = datetime(2025, 1, 1, 8, 0, 0)
END_DATE = datetime(2025, 12, 31, 18, 0, 0)
RANDOM_SEED = 42

# --- BUSINESS CALENDAR ---
WORK_START = 8  # 08:00
WORK_END = 18   # 18:00
PROB_OVERTIME = 0.05 # 5% chance of events happening outside business hours

# --- 4 ZONES (Exact Boundaries) ---
DRIFT_1 = 12500
DRIFT_2 = 25000
DRIFT_3 = 37500

class RealisticNonLinearGenerator:
    def __init__(self):
        self.log_dir = Path(__file__).parent / "logs"
        if not self.log_dir.exists(): self.log_dir.mkdir()
        
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        self.events = []
        
        # Calculate case spacing to fill the year
        total_seconds = (END_DATE - START_DATE).total_seconds()
        self.seconds_per_case = total_seconds / NUM_CASES

    def _get_case_start_time(self, i):
        # Linear distribution
        base_seconds = i * self.seconds_per_case
        # Add jitter (+/- 2 hours)
        jitter = random.randint(-7200, 7200)
        start_time = START_DATE + timedelta(seconds=base_seconds + jitter)
        return self._adjust_to_business_hours(start_time)

    def _adjust_to_business_hours(self, dt):
        """Ensures timestamp starts within working hours unless overtime."""
        if random.random() < PROB_OVERTIME:
            return dt

        # 1. Handle Weekends (Sat=5, Sun=6)
        while dt.weekday() >= 5:
            dt += timedelta(days=1)
            dt = dt.replace(hour=WORK_START, minute=random.randint(0, 59))

        # 2. Handle Night/Morning
        if dt.hour < WORK_START:
            dt = dt.replace(hour=WORK_START, minute=random.randint(0, 59))
        elif dt.hour >= WORK_END:
            dt += timedelta(days=1)
            dt = dt.replace(hour=WORK_START, minute=random.randint(0, 59))
            # Re-check weekend after adding a day
            while dt.weekday() >= 5:
                dt += timedelta(days=1)
                dt = dt.replace(hour=WORK_START, minute=random.randint(0, 59))
        
        return dt

    def _add_task_duration(self, current_time, min_mins=5, max_mins=120):
        """Adds duration and handles rollover to next day/week."""
        duration = random.randint(min_mins, max_mins)
        new_time = current_time + timedelta(minutes=duration)
        
        # If we are in overtime mode, just return the time
        if random.random() < PROB_OVERTIME:
            return new_time

        # Check if work day ended
        if new_time.hour >= WORK_END:
            # Move to next day morning
            overhead_minutes = new_time.minute
            new_time = new_time + timedelta(days=1)
            new_time = new_time.replace(hour=WORK_START, minute=overhead_minutes)
        
        # Check if it became a weekend
        while new_time.weekday() >= 5:
            new_time += timedelta(days=1)
            new_time = new_time.replace(hour=WORK_START, minute=random.randint(0, 30))
            
        return new_time

    def generate(self):
        print(f"Generating {NUM_CASES} cases (REALISTIC NON-LINEAR)...")
        
        for i in tqdm(range(NUM_CASES)):
            case_id = f"PAY-NL-{i:05d}"
            start_time = self._get_case_start_time(i)
            self._generate_single_trace(i, case_id, start_time)

        df = pd.DataFrame(self.events)
        df.sort_values(by=['case_id', 'timestamp'], inplace=True)
        
        pkl_path = self.log_dir / f"{OUTPUT_FILENAME}.pkl"
        xcl_path = self.log_dir / f"{OUTPUT_FILENAME}.xlsx"
        df.to_pickle(pkl_path)
        df.to_excel(xcl_path)
        
        print("\nGround Truth Table:")
        print(f"Zone 1 (0-12.5k): Agile Baseline (Full Parallelism)")
        print(f"Zone 2 (12.5k-25k): Hidden Constraint (R1: Timesheet->Budget)")
        print(f"Zone 3 (25k-37.5k): The Silo (R2: XOR Split D|E|F)")
        print(f"Zone 4 (37.5k-End): The Conflict (R1: Budget->Timesheet)")
        print(f"Saved to {pkl_path}")

    def _generate_single_trace(self, i, case_id, t):
        is_contractor = random.random() < 0.60
        bulletin_type = "Contractor" if is_contractor else "Permanent"
        
        # --- DETERMINE ZONE ---
        if i < DRIFT_1:
            phase = 1
            zone_label = "Zone 1: Agile Baseline"
            drift_desc = "Full Parallelism"
        elif i < DRIFT_2:
            phase = 2
            zone_label = "Zone 2: Hidden Constraint"
            drift_desc = "R1 Partial Order"
        elif i < DRIFT_3:
            phase = 3
            zone_label = "Zone 3: The Silo"
            drift_desc = "R2 XOR Split"
        else:
            phase = 4
            zone_label = "Zone 4: The Conflict"
            drift_desc = "R1 Reverse Constraint"

        def add(act, min_dur=10, max_dur=60):
            nonlocal t
            t = self._add_task_duration(t, min_dur, max_dur)
            self.events.append({
                'case_id': case_id,
                'activity': act,
                'timestamp': t,
                'bulletin_type': bulletin_type,
                'zone_label': zone_label,
                'drift_type': drift_desc,
                'case_index': i
            })

        add("Payroll Run Initiated", 5, 15)

        if bulletin_type == "Permanent":
            # CONTROL GROUP: Stable
            add("Retrieve Master Data", 5, 30)
            add("Calculate Base Salary", 10, 45)
            add("Generate Payslip", 5, 15)
            add("Execute Wire Transfer", 60, 180)
            
        else:
            # CONTRACTOR FLOW (The Drifting Process)
            add("Receive Invoice", 10, 60)

            # === REGION 1: OPS ===
            # A=Rate, B=Timesheet, C=Budget
            act_a, act_b, act_c = "Verify Rate", "Check Timesheet", "Validate Budget"
            
            if phase == 1 or phase == 3:
                # Full Parallel (Baseline & Zone 3)
                acts = [act_a, act_b, act_c]
                random.shuffle(acts)
                for a in acts: add(a, 20, 60)
            elif phase == 2:
                # Zone 2: Partial Order (B -> C fixed, A floats)
                # Static Miner Trap: It learned in Z1 that C->B is possible.
                seq = [act_b, act_c]
                seq.insert(random.choice([0, 1, 2]), act_a)
                for a in seq: add(a, 20, 60)
            elif phase == 4:
                # Zone 4: Reverse Constraint (C -> B fixed, A floats)
                # Static Miner Trap: Combined with Z2, B->C and C->B creates a Cycle.
                seq = [act_c, act_b]
                seq.insert(random.choice([0, 1, 2]), act_a)
                for a in seq: add(a, 20, 60)

            # === REGION 2: COMPLIANCE ===
            # D=Tax, E=Social, F=Legal
            act_d, act_e, act_f = "Check Tax", "Verify Social", "Legal Review"
            
            if phase == 3:
                # Zone 3: XOR Split (Only one)
                # Static Miner Trap: Learned in Z1 that D,E,F are mandatory. Fitness Crash.
                add(random.choice([act_d, act_e, act_f]), 30, 90)
            else:
                # All other zones: Full Parallel
                acts = [act_d, act_e, act_f]
                random.shuffle(acts)
                for a in acts: add(a, 30, 90)

            # === REGION 3: PAYMENT ===
            # G=Batch, H=Sign, I=Bank
            act_g, act_h, act_i = "Create Batch", "Treasury Sign", "Bank Confirm"
            
            # Always Parallel for consistency (Control Variable)
            acts = [act_g, act_h, act_i]
            random.shuffle(acts)
            for a in acts: add(a, 45, 120)

            add("Payroll Cycle Closed", 1, 5)

if __name__ == "__main__":
    RealisticNonLinearGenerator().generate()