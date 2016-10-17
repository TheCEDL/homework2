##Problem 5
有用baseline的結果會比沒有baseline更快收斂，如圖中，紅色的現為有用baseline的結果，大約在40個iteration收斂，比沒有用baseline的55個iteration更快
![image](https://github.com/ph81323/homework2-1/log.png)
原因可能為baseline應該是在當時的狀態下所有路徑未來期望reward的平均值，如果去掉未來的平均值，則表示可以單純只看這次的reward去做update，我們猜想這樣子的update方法較能沿著正確的路徑找到最好的policy

##Problem 6
做normalize的原因是因為，advantage裡的每一項是 t=0~T, t=1~T, t=2~T...等，scale並不一樣，所以要做normalize