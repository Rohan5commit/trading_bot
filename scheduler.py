"""
Daily Scheduler & Wi-Fi Checker

Logic:
1.  Runs continuously (or can be scheduled).
2.  Checks date: Has the bot run for TODAY already?
3.  Checks time: Is it between 08:00 and 00:00 (Midnight)?
4.  Checks Wi-Fi: Is internet available?
5.  If all YES -> Run `main.py`.
6.  If NO -> Sleep 5 minutes and retry.
"""
import time
import os
import sys
import logging
import subprocess
import socket
import json
import sqlite3
import yaml
from datetime import datetime, time as dtime, timezone, timedelta
from utils import get_sgt_now

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
START_TIME = dtime(8, 0)   # 8:00 AM
END_TIME = dtime(23, 59)   # Almost Midnight
CHECK_INTERVAL = 300       # 5 minutes
SCRIPT_TO_RUN = os.path.join(BASE_DIR, "main.py")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATE_FILE = os.path.join(BASE_DIR, "scheduler_state.json")
LOCK_FILE = os.path.join(BASE_DIR, "scheduler.lock")
CONFIG_FILE = os.path.join(BASE_DIR, "config.yaml")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - SCHEDULER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "scheduler.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def is_connected():
    """Check for internet connectivity by pinging Google DNS"""
    try:
        # Connect to 8.8.8.8 on port 53 (DNS) - fast and reliable
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def _load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as handle:
            return json.load(handle)
    except Exception:
        return {}

def _save_state(state):
    try:
        with open(STATE_FILE, "w") as handle:
            json.dump(state, handle, indent=2)
    except Exception as exc:
        logger.warning(f"Failed to persist scheduler state: {exc}")

def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def _acquire_lock():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as handle:
                payload = json.load(handle)
            pid = int(payload.get("pid", 0))
            if pid and _pid_alive(pid):
                logger.error("Scheduler already running (pid=%s). Exiting.", pid)
                return False
        except Exception:
            pass
    payload = {
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat()
    }
    try:
        with open(LOCK_FILE, "w") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as exc:
        logger.warning(f"Failed to write scheduler lock: {exc}")
    return True

def has_run_today():
    """
    Check if we've already completed a successful run for the current SGT date.

    IMPORTANT:
    - We MUST run at least once per valid SGT day to ingest fresh prices; otherwise the
      DB market date never advances and the bot can get stuck after weekends.
    - A run is only considered successful once the pipeline wrote
      `results/email_sent_<market_date>.ok` (marker), so we can retry if email fails.
    """
    state = _load_state()
    now_sgt = get_sgt_now()
    today_sgt = now_sgt.strftime("%Y%m%d")
    return (state.get("last_success_sgt_date") == today_sgt) and (state.get("last_status") == "success")

def _load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}

def get_latest_market_date_str():
    """Return latest `prices.date` (YYYYMMDD) or None."""
    cfg = _load_config()
    db_rel = ((cfg.get("data") or {}).get("cache_path")) or "./data/trading_bot.db"
    db_path = db_rel if os.path.isabs(db_rel) else os.path.join(BASE_DIR, db_rel)
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return None
        # `date` is stored as YYYY-MM-DD in SQLite.
        return str(row[0]).replace("-", "")
    except Exception as exc:
        logger.warning("Failed to read latest market date: %s", exc)
        return None

def run_pipeline():
    """Execute the main trading pipeline"""
    logger.info("Conditions met. Starting Trading Bot Pipeline...")
    try:
        # Run main.py using the same python interpreter
        result = subprocess.run(
            [sys.executable, SCRIPT_TO_RUN],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Pipeline completed successfully.")
            logger.info(result.stdout)
            return True
        else:
            logger.error("Pipeline failed!")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Failed to launch pipeline: {e}")
        return False

def get_sgt_time():
    """Get current time in Singapore (UTC+8)"""
    return get_sgt_now()

def main_loop():
    logger.info("Scheduler started. Monitoring for 8am start time (Singapore Time)...")
    if not _acquire_lock():
        return
    
    while True:
        try:
            # Get current time in SGT
            now_sgt = get_sgt_time()
            current_time = now_sgt.time()
            current_day = now_sgt.weekday()  # 0=Monday, 6=Sunday
            today_str = now_sgt.strftime('%Y%m%d')

            # 1. Day of Week Check
            # User requested: Don't run on Sunday (6) or Monday (0)
            # Because US Market is closed Sat/Sun, so Mon morning SGT (Sun night US) and Sun morning SGT (Sat night US) have no data.
            # We run Tue-Sat SGT (processing Mon-Fri US data).
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            logger.debug(f"Current day is {day_names[current_day]} (day {current_day})")

            if current_day in [0, 6]: # 0=Mon, 6=Sun
                day_name = "Sunday" if current_day == 6 else "Monday"
                # Log only once per hour to avoid spam
                if current_time.minute == 0:
                    logger.info(f"Today is {day_name} (SGT). Market closed. Skipping run.")
                time.sleep(3600) # Sleep 1 hour
                continue
            else:
                logger.debug(f"Today is {day_names[current_day]} - valid trading day.")

            # 2. Check if already ran today
            # Note: has_run_today uses local file timestamps. Ideally we check against SGT date.
            # But for valid days, we just rely on the file existence for 'today'.
            logger.debug(f"Checking if already ran today. Current day: {current_day}, Today date: {today_str}")
            if has_run_today():
                logger.info("Bot has already completed its run for the latest market date. Waiting...")
                time.sleep(3600)
                continue
            else:
                logger.debug("Bot has not run today yet. Proceeding with checks.")

            # 3. Check Time Window (8am - 12am SGT)
            logger.debug(f"Checking time window. Start: {START_TIME}, End: {END_TIME}, Current: {current_time}")
            if not (START_TIME <= current_time <= END_TIME):
                # Only log occasionally
                if current_time.minute % 30 == 0:
                    logger.info(f"Outside execution window ({START_TIME}-{END_TIME}). Current SGT time: {current_time.strftime('%H:%M')}.")
                time.sleep(CHECK_INTERVAL)
                continue
            else:
                logger.debug("Within execution window. Proceeding with Wi-Fi check.")

            # 4. Check Wi-Fi
            wifi_status = is_connected()
            logger.debug(f"Wi-Fi connection status: {wifi_status}")
            if not wifi_status:
                logger.warning("Inside window but NO INTERNET. Retrying in 5 minutes...")
                time.sleep(CHECK_INTERVAL)
                continue
            else:
                logger.debug("Wi-Fi connection confirmed. Proceeding to run pipeline.")

            # 5. All Green - RUN!
            market_date_str = get_latest_market_date_str() or today_str
            logger.info(
                "All conditions met. Starting pipeline run for SGT=%s (%s), market_date=%s",
                today_str,
                day_names[current_day],
                market_date_str,
            )
            state = _load_state()
            state["last_attempt_at"] = now_sgt.isoformat()
            state["last_status"] = "running"
            _save_state(state)

            success = run_pipeline()

            # Consider the run successful only if the pipeline wrote the email-sent marker
            # for the latest market date.
            market_date_str = get_latest_market_date_str() or market_date_str
            marker = os.path.join(RESULTS_DIR, f"email_sent_{market_date_str}.ok")
            if success and os.path.exists(marker):
                logger.info("Daily run finished and email marker exists (%s).", os.path.basename(marker))
                state["last_run_market_date"] = market_date_str
                state["last_success_sgt_date"] = today_str
                state["last_status"] = "success"
            else:
                if success and not os.path.exists(marker):
                    logger.error("Pipeline exited successfully but email marker missing (%s). Will retry.", os.path.basename(marker))
                else:
                    logger.error("Run failed. Will retry in 5 minutes (unless fixed).")
                state["last_status"] = "failed"
            state["last_finished_at"] = get_sgt_time().isoformat()
            _save_state(state)

            time.sleep(CHECK_INTERVAL)
        except Exception as exc:
            logger.exception(f"Scheduler loop error: {exc}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")
