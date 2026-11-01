import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

hot = 0.6
cold = 0.5
model = joblib.load("digital_twin/gp_model.joblib")
times, hots, colds, mixing_indexs, uncertainties = [],[],[],[],[]
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for second in range(30):
    hot =  hot + np.random.normal(0,0.05)
    cold = cold + np.random.normal(0,0.01)
    hot = np.clip(hot, 0.4, 2.0)
    cold = np.clip(cold,0.2,0.6)
    live = pd.DataFrame({"Hot_Inlet": [hot], "Cold_Inlet" : [cold]})
    mixing_index,uncertainity = model.predict(live, return_std = True)
    mixing_index = mixing_index[0]
    uncertainity = uncertainity[0]
    print(f't = {second}s hot = {hot:.2f} m/s cold= {cold :.2f} m/s  mixing_percentage = {mixing_index*100 :.3f} +- {uncertainity*100 :.3f}')
    if mixing_index < 0.5:
        print("Poor mixing")
        hot_options = np.linspace(0.4,2.0,100)
        trial = pd.DataFrame({"Hot_Inlet" : hot_options , "Cold_Inlet" : cold})
        trial_run =  model.predict(trial)
        good_options = hot_options[trial_run >=0.55]
        if len(good_options) > 0:
            print(f" raise the hot inlet to {good_options[0] :.2f} m/s")
        else:
            print('need to adjust cold options also')
    if uncertainity > 0.02:
        print("Prediction is less reliable")
    times.append(second)
    hots.append(hot)
    colds.append(cold)
    mixing_indexs.append(mixing_index)
    uncertainties.append(uncertainity)
    ax1.clear()
    ax1.plot(times, hots, label="hot")
    ax1.plot(times, colds, label="cold")
    ax1.set_ylabel("Inlet speed (m/s)")
    ax1.legend()
    m = np.array(mixing_indexs)
    u = np.array(uncertainties)
    ax2.clear()
    ax2.set_ylim(0.4, 0.7)
    ax2.plot(times, m, color="black", label="predicted mixing")
    ax2.fill_between(times, m - 2*u, m + 2*u, alpha=0.3, label="±2 std")
    ax2.axhline(0.5, color="red", linestyle="--", label="alarm limit")
    ax2.set_ylabel("Mixing index")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    plt.pause(1)

plt.ioff()
plt.savefig("digital_twin/twin_demo.png", dpi = 200)
plt.show()
