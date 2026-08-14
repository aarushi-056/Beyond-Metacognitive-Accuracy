import pandas as pd
import hddm
model = hddm.load("model4.hddm")
v_hc_high = model.nodes_db.node["v(HC_a_high)"].trace()
v_hc_low = model.nodes_db.node["v(HC_a_low)"].trace()
v_ssd_high = model.nodes_db.node["v(SSD_a_high)"].trace()
v_ssd_low = model.nodes_db.node["v(SSD_a_low)"].trace()
a_hc_high = model.nodes_db.node["a(HC_a_high)"].trace()
a_hc_low = model.nodes_db.node["a(HC_a_low)"].trace()
a_ssd_high = model.nodes_db.node["a(SSD_a_high)"].trace()
a_ssd_low = model.nodes_db.node["a(SSD_a_low)"].trace()	
results = []

p = (v_hc_high > v_hc_low).mean()
results.append( [ "HC high confidence drift > HC low confidence drift", 
                 p ])
p = (v_ssd_high > v_ssd_low).mean()
results.append( [ "SSD high confidence drift > SSD low confidence drift",
                 p ])
p = (a_hc_high > a_hc_low).mean()
results.append([ "SSD high confidence boundary > SSD low confidence boundary",
				 p ])

hc_effect = a_hc_high - a_hc_low
ssd_effect = a_ssd_high - a_ssd_low
p = (hc_effect > ssd_effect).mean()
results.append(["HC confidence effect > SSD confidence effect",
                p])
results = pd.DataFrame( 
    results, 
    columns = ["Hypothesis", "Posterior probability"
	]
)
print (results)
results.to_csv("model4_posterior_hypotheses.csv", index = False)
print ("bazinga")