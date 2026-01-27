## Set of Pitches $\mathcal{P}$
We define the set of pitches $\mathcal{P} = \mathbb{N}\mod p$ with $p$ as the number of pitches. All can be extended to more pitches. Here $12 = \mathcal{|P|}=card(\mathcal{P})$ for simplicity.
# Chroma
- #Chroma:  function $\mathcal{P} \rightarrow \{0,1\}$. It is a fixed and rolling representation of a binary information over the twelve-pitches. Example : $(0,1,2,3,4,5,6,7,8,9,10,11)\rightarrow(1,0,1,0,1,1,0,1,0,1,0,1)$
- #Chroma-objects: objects that can be defined entirely with a **Chroma**
- A **Chroma** allows repetition, inversion, access to boolean relative to a value of $\mathcal{P}$, access to a subset of itself.
- Can be defined by the largest subset $P \subset \mathcal{P}$ where $Chr(P) = \{1\}$
	- $P = Chr^{-1}(\{1\})$
	- $P$ is the Pitch representation of the Chroma: $(0,2,4,5,7,9,11)$
	- We can define a function $x \rightarrow P[x]$
## ChromaPitch
Chroma for which all values are 0 except for one member of $\mathcal{P}$
- Methods
	- Addition: similar to shift
	- same methods as Chroma
## ChromaInterval
Representation of an Interval as a Chroma.
Strictly identical to a ChromaPitch, with different methods.
- Methods:
	- symmetrical : find the symmetrical interval.
	- Addition between two intervals : similar to shift

=> ChromaPitch and ChromaInterval are useless unless we want to prepare a Chroma to be used by other functions needing intervals or Pitches.
=> There could be a function extracting all intervals from a chroma, and a function extracting all the pitches from a chroma.
=> The advantage if you take a chord (3 ones) and tranform it to pitches, you only need to unionize them to get the chord back. If it's a list of pitches, you need to recreate a list.

