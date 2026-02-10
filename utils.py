import os
from datetime import datetime
import pytz

def get_sgt_now():
    """Returns the current datetime in Singapore Time (SGT)."""
    sgt_tz = pytz.timezone('Asia/Singapore')
    return datetime.now(sgt_tz)

def get_sgt_date_str():
    """Returns the current date in SGT as YYYYMMDD string."""
    return get_sgt_now().strftime('%Y%m%d')

def get_sgt_time_str():
    """Returns the current time in SGT as HH:MM:SS string."""
    return get_sgt_now().strftime('%H:%M:%S')
