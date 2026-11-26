import json
from decision import decide
payload=dict(distance_m=140,ambulance_speed_kmh=55,queue_cars=1,density_pct=20,flow_speed_kmh=45,weather_code=0,hour=11,weekday=1,gps_age_s=1)
for name, sample in [('Fresh GPS',payload),('Stale GPS',{**payload,'gps_age_s':6})]:
    print(name, json.dumps(decide(sample,route_confirmed=True,sensors_ok=True,hazard=False),indent=2))
