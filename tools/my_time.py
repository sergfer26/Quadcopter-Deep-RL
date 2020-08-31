from datetime import datetime
from datetime import datetime, timezone
import pytz

def my_date():
    #naive_dt = datetime.now()
    #naive_utc_dt = datetime.utcnow()
    #utc_dt = datetime.now(timezone.utc)
    # dt = utc_dt.astimezone()
    tz = pytz.timezone('America/Mexico_City')
    mexico_now = datetime.now(tz)
    mes = mexico_now.month
    dia = mexico_now.day
    hora = mexico_now.hour
    minuto = mexico_now.minute
    return{'month': mes, 'day': dia, 'hr': hora, 'min': minuto}