"""PR8: conservative simulation gate implementing PR7 recommendations.
No hardware, network or traffic-light commands. Never use for live control.
"""
import math
from eta import predict_eta

LIMITS = {'distance_m': (70,650), 'ambulance_speed_kmh': (15,75),
          'queue_cars': (0,17), 'density_pct': (10,95), 'flow_speed_kmh': (8,65),
          'weather_code': (0,2), 'hour': (0,23), 'weekday': (0,6)}
DISCRETE = {'queue_cars','weather_code','hour','weekday'}

def decide(payload, *, route_confirmed=False, sensors_ok=False, hazard=True, override=False):
    result = {'mode':'REJECT', 'reason':'', 'prediction':None, 'command_sent':False}
    def finish(mode, reason):
        return {**result, 'mode':mode, 'reason':reason}
    def valid(x):
        return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
    age=payload.get('gps_age_s')
    if not valid(age) or not 0 <= age <= 5:
        return finish('REJECT','DP-10: invalid/stale GPS; dispatcher')
    if not sensors_ok:
        return finish('REJECT','DP-09: sensor validity not confirmed')
    for k,(lo,hi) in LIMITS.items():
        v=payload.get(k)
        if not valid(v) or not lo<=v<=hi or (k in DISCRETE and v!=int(v)):
            return finish('REJECT',f'Invalid/out-of-domain feature: {k}')
    try:
        p=predict_eta(payload); result['prediction']=p
        t=p['predicted_s']; score=p['confidence_score']; lo,hi=p['interval_s']
        if not all(valid(x) for x in (t,score,lo,hi)) or min(t,lo,hi)<0 or not 0<=score<=1 or lo>hi:
            return finish('REJECT','Invalid model output')
    except Exception:
        return finish('REJECT','Model error; dispatcher')
    if score<.5:
        return finish('REJECT','Score below 0.50')
    if override or hazard or not route_confirmed:
        return finish('REVIEW','Human control / unsafe or unconfirmed route')
    if not lo<=t<=hi or score<=.90 or lo<15<=hi:
        return finish('REVIEW','Uncertain or inconsistent prediction')
    if lo>=15:
        return finish('WAIT','Too early; refresh data')
    if t<15 and hi<15:
        return finish('SIMULATED_GREEN','DP-04: recommendation only')
    return finish('REVIEW','Unresolved case')
