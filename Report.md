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
不過從圖中(Fig 1)我們還是可以看出來，雖然一開始有baseline(紅)的return比較小，但是他還是可以花比較少的iteration收斂<br>
![Fig. 1](https://github.com/CEDL739/homework2/blob/master/reward(b).png)<br>
　　　　　　　　　　　　　　　　　　　　　　**Fig .1** With/without baseline <br>
但是我們覺得最重要的是最一開始的return，如果一開始的return很高的話，就可以在很少的iteration內收斂<br>

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


