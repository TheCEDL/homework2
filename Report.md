## Homework2 - Policy Gradient
Please complete each homework for each team, and <br>
mention who contributed which parts in your report.

## Member
* captain - <a href="https://github.com/maolin23?tab=repositories">林姿伶</a>: 104062546
* member -  <a href="https://github.com/hedywang73?tab=repositories">汪叔慧</a>: 104062526
* member - <a href="https://github.com/yenmincheng0708?tab=repositories">嚴敏誠</a>: 104062595

` Contribution `
```
姿伶 and 叔慧 discussed the homework.
Finally, 姿伶 typed the final report on github.
```

## Problem 5
```
Replacing line
baseline = LinearFeatureBaseline(env.spec)
with
baseline = None
can remove the baseline.
Modify the code to compare the variance and performance before and after adding baseline. 
Then, write a report about your findings. (with figures is better)
```
理論上，利用減掉baseline這個動作，可以降低variance，於是可以得到比較好的predict結果<br>
但可能因為這個task太簡單，所以在 return 上，有沒有baseline對結果差異沒有那麼大<br>
不過從圖中我們還是可以看出來，雖然一開始有baseline(紅)的return比較小，但是他還是可以花比較少的iteration收斂<br>
![Fig. 1](https://github.com/CEDL739/homework2/blob/master/reward(b).png)<br>
　　　　　　　　　　　　　　　　　　　　　　**Fig .1** With/without baseline

## Problem 6
```
In function process_paths of class PolicyOptimizer, why we need to normalize the advantages? 
i.e., what's the usage of this line:
p["advantages"] = (a - a.mean()) / (a.std() + 1e-8)
Include the answer in your report.
```
因為在前面為了降低varience，減掉了baseline，這個動作使得整個model的distribution偏移<br>
所以在這邊要對a做normalize將model拉回來<br>


