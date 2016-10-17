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
下面這兩張圖，是分別在有或沒有baseline的情況下各跑10次的結果<br>
中間的曲線代表的是十次平均的結果，上及下的值分別代表的是在這10次當中，該iteration的最大最小值<br>
從這兩個結果看起來，我們覺得或許是因為這個task太簡單，所以不管有沒有baseline，對於學習的影響不大<br>
![Fig. 1](https://github.com/CEDL739/homework2/blob/master/reward_with.png)<br>
　　　　　　　　　　　　　**Fig .1** With baseline<br>
![Fig. 2](https://github.com/CEDL739/homework2/blob/master/reward_without.png)<br>
　　　　　　　　　　　　　**Fig .2** Without baseline<br>
但是我們覺得最重要的是最一開始的initial weight，如果一開始的return很高的話，就可以在很少的iteration內收斂<br>

## Problem 6
```
In function process_paths of class PolicyOptimizer, why we need to normalize the advantages? 
i.e., what's the usage of this line:
p["advantages"] = (a - a.mean()) / (a.std() + 1e-8)
Include the answer in your report.
```
對 advantage function 進行normalize是因為，當我們計算accumulated reward時，是進行discounted reward計算。<br>
每個當下的reward都會乘上一個discounted factor，並且這個factor是隨著stage的進行呈現exponentially discounted。<br>
因此，比較後面的stage的action會因為乘上這個discounted factor，而使得學習上比較沒有效率。<br>
所以如果對整個時間點的advantage進行normalize，就可以使得整段時間點的每個stage影響比較平均。<br>


