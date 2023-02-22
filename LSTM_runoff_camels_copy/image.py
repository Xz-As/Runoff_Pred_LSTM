from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
from pandas import read_csv as csv
import pickle
# Plot results
fig = plt.figure(figsize=(15, 10))
#plt.figure(1)
cnt = 0

matplotlib.rc('xtick', labelsize=9)
matplotlib.rc('ytick', labelsize=9)
def draw_(basin, cnt):
    ax = fig.add_subplot(2, 3, cnt)
    f = csv(basin+'_output.txt', sep='\s+', header=0)
    obs = f['Obs']/1000
    pred = f['Pred']/1000
    #print(len(obs), len(pred))
    print(min(obs), max(obs), max(pred))
    mx = max((max(obs), max(pred))) * 1.01
    ax.axis((0, mx, 0, mx))
    ax.plot((0, mx), (0, mx), 'k-')
    plt.xlabel('observation')
    plt.ylabel('prediction')
    ax.set_title('basin:' + basin)
    ax.scatter(obs, pred, marker = '.')
    ax.xaxis.set_major_locator(plt.MultipleLocator(int(mx / 400) * 100))
    ax.yaxis.set_major_locator(plt.MultipleLocator(int(mx / 400) * 100))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%dm'))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%dm'))
    #plt.show()

    '''start_date = 0
    end_date = int(max(obs) * 1.01)	
    date_range =[str(i) for i in range(start_date, end_date, )]

    #if not cal:
    #    ax.plot(date_range, obs, label=f"observation, NSE = {nse:.3f}")
    ax.plot(date_range, preds, label=f"prediction")
    ax.legend()
    ax.set_title(f"Basin {basin}")
    # ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Date")
    ax.set_xticks([])
    _ = ax.set_ylabel("Discharge (mm/d)")'''

basins = ('01022500', '01052500', '01031500', '01047000', '01054200', '01057000')
for i in basins:
    cnt += 1
    draw_(i, cnt)
plt.subplots_adjust(wspace = 0.33, hspace = 0.33)
plt.savefig(fname = 'l_runoff.svg', format='svg')
#plt.show()
