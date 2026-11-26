"""Boundary tests with stub predictions + real model smoke test."""
import unittest
from unittest.mock import patch
from decision import decide
from eta import mae, baseline, coverage
P=dict(distance_m=140,ambulance_speed_kmh=55,queue_cars=1,density_pct=20,flow_speed_kmh=45,weather_code=0,hour=11,weekday=1,gps_age_s=1)
FLAGS=dict(route_confirmed=True,sensors_ok=True,hazard=False)
def pred(t=10,lo=7,hi=12,score=.93):
    return dict(predicted_s=t,interval_s=[lo,hi],confidence_score=score,command_sent=False)
class Policy(unittest.TestCase):
    def check(self, expected, output=None, changes=None, flags=None):
        with patch('decision.predict_eta',return_value=output or pred()):
            r=decide({**P,**(changes or {})},**{**FLAGS,**(flags or {})})
        self.assertEqual(r['mode'],expected); self.assertFalse(r['command_sent'])
    def test_green(self): self.check('SIMULATED_GREEN')
    def test_wait(self): self.check('WAIT',pred(20,15,24))
    def test_threshold_interval(self): self.check('REVIEW',pred(12,9,15))
    def test_score_boundary(self): self.check('REVIEW',pred(score=.90))
    def test_low_score(self): self.check('REJECT',pred(score=.49))
    def test_inconsistent(self): self.check('REVIEW',pred(13,5,10))
    def test_gps_5(self): self.check('SIMULATED_GREEN',changes={'gps_age_s':5})
    def test_gps_6(self): self.check('REJECT',changes={'gps_age_s':6})
    def test_missing(self): self.check('REJECT',changes={'queue_cars':None})
    def test_out_of_range(self): self.check('REJECT',changes={'distance_m':900})
    def test_override(self): self.check('REVIEW',flags={'override':True})
    def test_hazard(self): self.check('REVIEW',flags={'hazard':True})
    def test_sensor(self): self.check('REJECT',flags={'sensors_ok':False})
    def test_route(self): self.check('REVIEW',flags={'route_confirmed':False})
    def test_model_error(self):
        with patch('decision.predict_eta',side_effect=RuntimeError):
            self.assertEqual(decide(P,**FLAGS)['mode'],'REJECT')
    def test_real_model(self):
        r=decide(P,**FLAGS); self.assertIsNotNone(r['prediction']); self.assertFalse(r['command_sent'])
if __name__=='__main__':
    print(f'PR6 rerun: MAE={mae:.2f}; baseline={baseline:.2f}; coverage={coverage:.1%}',flush=True)
    unittest.main(verbosity=2)
