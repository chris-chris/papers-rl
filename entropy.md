$$H(P) = H(x) = E_{X \sim P}[I(x)] = -E_{X \sim P} [logP(x)]$$ 

$$H(X) = - \sum_{k=1}^K p(X=k)logp(X=k) $$

$$p(X=1)= 0.5$$

$$p(X=0)= 0.5$$

$$-[p(X=0)logp(X=0) + p(X=1)logp(X=1)]$$
$$= -[0.5log0.5 + 0.5log0.5]$$
$$= -[0.5 * -0.6931471805599453 * 2] $$
$$= 0.693147180559945 $$
$$ $$


$$p(X=1)= 1$$

$$p(X=0)= 0$$

$$-[p(X=0)logp(X=0) + p(X=1)logp(X=1)]$$
$$= -[1log1 + 0log0] \\= -[0 + 0] \\= 0 $$
$$ $$

$$D_{KL}(P||Q) = \sum_iP(i)log\frac{P(i)}{Q(i)} 
\\= \sum_iP(i)log{P(i)} - \sum_iP(i)log{Q(i)} 
\\= -H(P) + H(P,Q)$$
