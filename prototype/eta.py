"""Original PR6 model (notebook cells 2, 4, 6); training is reproducible."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
rng=np.random.default_rng(42)
n=1200
df=pd.DataFrame({"distance_m":rng.uniform(70,650,n),"ambulance_speed_kmh":rng.uniform(15,75,n),"queue_cars":rng.integers(0,18,n),"density_pct":rng.uniform(10,95,n),"flow_speed_kmh":rng.uniform(8,65,n),"weather_code":rng.choice([0,1,2],n,p=[.65,.27,.08]),"hour":rng.integers(0,24,n),"weekday":rng.integers(0,7,n)})
v=np.maximum(8,.67*df.ambulance_speed_kmh+.33*df.flow_speed_kmh)
df["actual_time_s"]=(df.distance_m/(v/3.6)+.55*df.queue_cars+.035*df.density_pct+3*df.weather_code+np.where(df.hour.between(7,9)|df.hour.between(16,19),3.5,0)+rng.normal(0,2,n)).clip(lower=2).round(2)

features=[c for c in df if c!="actual_time_s"]
X_train,X_test,y_train,y_test=train_test_split(df[features],df.actual_time_s,test_size=.2,random_state=42)
models={}
for name,loss,alpha in [("mean","squared_error",.9),("low","quantile",.1),("high","quantile",.9)]:
    model=GradientBoostingRegressor(loss=loss,alpha=alpha,n_estimators=170,max_depth=3,learning_rate=.06,random_state=42)
    model.fit(X_train,y_train)
    models[name]=model
pred=models["mean"].predict(X_test)
low=models["low"].predict(X_test)
high=models["high"].predict(X_test)
lo=np.minimum(low,high); hi=np.maximum(low,high)
mae=mean_absolute_error(y_test,pred)
baseline=mean_absolute_error(y_test,X_test.distance_m/(np.maximum(X_test.ambulance_speed_kmh,1)/3.6))
coverage=np.mean((y_test>=lo)&(y_test<=hi))

def predict_eta(payload):
    gps_age=payload.get("gps_age_s")
    if gps_age is None or gps_age<0 or gps_age>5:
        return {"decision":"ESCALATE_STALE_GPS", "predicted_s":None, "confidence_score":None}
    if any(k not in payload or pd.isna(payload[k]) for k in features):
        return {"decision":"ESCALATE_MISSING_DATA", "predicted_s":None, "confidence_score":None}
    row=pd.DataFrame([{k:payload[k] for k in features}])
    t=float(models["mean"].predict(row)[0]); l=float(models["low"].predict(row)[0]); h=float(models["high"].predict(row)[0])
    l,h=min(l,h),max(l,h)
    score=float(np.clip(1-(h-l)/100,0,1))
    decision="RECOMMEND_GREEN" if t<15 and score>.90 else "NO_AUTO_GREEN"
    return {"predicted_s":round(t,2),"interval_s":[round(l,2),round(h,2)],"confidence_score":round(score,3),"decision":decision,"command_sent":False}

