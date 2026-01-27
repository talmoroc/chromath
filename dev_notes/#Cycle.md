### #Cycle

#### Definition
A cycle $\mathcal{C}$ is defined by a recurring step $s_c\in\mathcal{P}\setminus \{0\}$  and a starting point $c_0 \in\mathcal{P}$
$$
\forall n \in \mathcal{P}, \mathcal{C}(n) = c_0 + s_c\times n \pmod{\mathcal{|P|}}$$
#### Definition
For any cycle $C = C_{c_0, s_c}$ we define the symmetrical cycle $\overline{C}$ such that $\forall n\in\mathcal{P}: \overline{C}(n) = C(-n)$
###### Property : $\overline{C_{c_0, s_c}} = C_{c_0,card(\mathcal{P})-s_c}$ 
$$\begin{align*}
&1. &s_c \in \mathcal{P} \setminus \{0\} &\Rightarrow \left(card(\mathcal{P}) - s_c\right) \in \mathcal{P} \setminus \{0\}\\
\\
&2. &\forall n \in \mathcal{P}, \bar{C}(n) &= c_0 + (\mathcal{|P|} - s_c)n &\pmod{\mathcal{|P|}} 
\\ &&&= c_0 - s_c n + \mathcal{|P|}n&\pmod{\mathcal{|P|}}
\\ &&&= c_0 - s_c n  + \mathcal{|P|}&\pmod{\mathcal{|P|}}
\\ &&&= C(-n)
\end{align*}$$
###### Property : Any cycle and its symmetrical have the same image.

###### Property: the maximum periodicity of the cycle is $card(\mathcal{P})$ $$\begin{align*}
&\forall n \in \mathcal{P}, &C(n + \mathcal{|P|}) & = c_0 + s_c\times (n + \mathcal{|P|}) &\pmod{\mathcal{|P|}} 
\\&&&= c_0 + s_c n + s_c \mathcal{|P|}&\pmod{\mathcal{|P|}}
\\&&&= c_0 + s_c n &\pmod{\mathcal{|P|}}
\\&&&= C(n)
\end{align*}$$
- The step generates a periodicity $\pi_C$ relative to the number of pitches $\mathcal{|P|}$ :
 $$\begin{align*}
 &\text{1.  }\mathcal{|P|} = 0\pmod{s_c} &\iff &\mathcal{|P|} \text{ divisible par } s_c\\
 &&\iff& \pi_c = \frac{\mathcal{|P|}}{s_c}\\

&\text{2. }\mathcal{|P|} \neq 0\pmod{s_c} &\iff &\mathcal{|P|} \text{ non divisible par } s_c\\
&&\iff&\pi_c = 12
\end{align*} 
$$

	**Note** This is actually an exceptional case because 12 is and has many divisors. In the general case if $\mathcal{|P|} = x$ (microtonal music...) we need to check in 2 steps :
			1. First by symmetry we consider the cycle $C^*$ such as $s^* = \text{min}(s_c, \mathcal{|P|}-s_c)$
			2. Is there $a\in\mathcal{P}$ such that $|\mathcal{P}| = as^*$ ? If yes, $\pi_c = \frac{\mathcal{|P|}}{s^*}$
			3. If not, we have $\mathcal{|P|} = as^* + r$
				1. Is there $a' \in \mathcal{|P|}$ such that $s^* = a'r$ ? If yes, $\pi_c = \frac{\mathcal{|P|}}{r}$
				2. If not, we have $\pi_c =  \mathcal{|P|}$.

#### Definition
- If $\pi_c < \mathcal{|P|}$ we have a _Partial Cycle_ : $\mathcal{C}(\mathcal{P}) \subsetneq \mathcal{P}$.
	- We need $s_c$ cycles with $c_0 \in \{0,1,...s_c - 1\}$ to cover all $\mathcal{P}$
	- Can be represented by a **Chroma**
	- However the Chroma doesn't give the order in which pitches appear in the $\mathcal{C}$ sequence.
- If $\pi_c = \mathcal{|P|}$ we have a _Complete Cycle_ : $\mathcal{C}(\mathcal{P}) = \mathcal{P}$.

### Scale
- 
