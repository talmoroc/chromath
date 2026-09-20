# Main objective

- have an algorithm that takes as input :
    1. A list of chords OR a tonality (root + scale from a selection)
    2. A desired sequence length
    2. Diverse meta-parameters to bias the generation

- and outputs :
    1. Missing chords in the sequence (several propositions)
    2. Information about the chords (analysis of the role/movement and evolution of meta-parameters)

- with two modes :
    1. Hands-free : the algorithm fills the gaps
    2. Iterative : the algorithm proposes several chords and the user chooses, which influences the next proposed chords.


# Main steps
### Data structures
#### Chroma

#### Degree
This is an array `[[Degree: int], [Semitone: int]]`

Now the questions (Example for a F major chord):
1) degrees of a chords ?
```python
# Scale:
[
    [0,1,2,3,4,5,6],
    [0,2,4,5,7,9,11]
]

# Chords: version 1
[
    [0,1,2],
    [0,5,9]
]

# Chords: version 2
[
    [0,2,4],
    [0,5,9]
]

"""
The goal being to place chords within a scale, the version 2 seems better. It represents chords as thirds. The difference can be easily made.

However to get this, the chord need to be built from a scale !
"""
```



2) ordered by semitone or degree ?
```python
# Ordered by semitone
[
    [4,0,2],
    [0,5,9]
]

# Ordered by degree
[
    [0,2,4],
    [5,9,0]
]

"""
It's better ordered by degree when iterating over the chord. index -> (degree, semitone) in their order
Makes similar chord more comparable.
"""

# Ordered by degree
[ # F scale
    [0,1,2,3,4,5,6],
    [5,7,9,10,0,2,4]
]
[ # C scale
    [0,1,2,3,4,5,6],
    [0,2,4,5,7,9,11]
]

# Ordered by semitone
[ # F scale
    [4,5,6,0,1,2,3],
    [0,2,4,5,7,9,10]
]
[ # C scale
    [0,1,2,3,4,5,6],
    [0,2,4,5,7,9,11]
]
"""
Both have their advantage when comparing two scales, and give different information.
When comparing objects that have the same number of degrees, the semitone ordering allow a chroma comparison [0,0,0,0,0,0,-1] -> F scale has the same tone as C scale except for the 11 becoming a 10. Alteration computation.
The degree ordering shows that the structure are similar :
[5,5,5,5,-7,-7,-7] with -7 mod 12 = 5. Same scale, with 5 semitones shift.

    Note :
    In the case of a scale, the [0,1,2,3,4,5,6] array is redundant.
    However, if you take a subset of a scale (interval, or degree), it become important.
    F chord = [0,2,4] of F scale

"""


# Combining both representations : F scale
[
    [0,  1,  2,  3,  4,  5,  6  ],
    [5,  7,  9,  10, 0,  2,  4  ],
    [4,  5,  6,  0,  1,  2,  3  ],
    [0,  2,  4,  5,  7,  9,  10 ]
]
# It seems it's just redundant information because it is a rotation. However, sometimes the ordering is non linear : when we have a chord with degrees over 7 for instance.

# C7sus2
[
    [0,  1,  4,  6  ], # Ordered by degree
    [0,  2,  7,  10 ],
    [0,  1,  4,  6  ], # Ordered by semitone
    [0,  2,  7,  10 ]
]

# C9 no3
[
    [0,  4,  6,  8  ], # Ordered by degree
    [0,  7,  10, 2  ],
    [0,  8,  4,  6  ], # Ordered by semitone
    [0,  2,  7,  10 ]
]
```

### Do we need degrees over 6 ?
**Rules**
- A 2d is a 9th if there is a 3d
- A 4th is a 11th if there is a 3d
- A 6th is a 13th if there is a 7th
- A 8th is always a 0th
- A 2d is a 9th is there is a 4th and a 7th
- A 2d is a 9th if there is a 7th and no third


**a. Rule 1**

>    - A 2d, 4th are always 9th and 11th.
>    - 6th is a 13th only if there is a 7th

**b. Rule 2**
    
>    - The chords are always built with stacked thirds + extensions that are not stacked thirds.
>    - If there is a stacked third with a higher degree over a non-stacked third (ex: if there is a 7th over a 6th) then it falls back to a higher degree (13th). Else it stays.
>    - The 5th and 8th are excluded from this rule.

**c. Rule 3: the good one ?**

>- The chords are build with stacked thirds + extensions that are not stacked thirds.
>- The fundamental and the fifth are constituent, if they are missing they do not change the interpretation of the chord.
>- The third and the 7th are essential to the color of the chord. If they are missing, they can be replaced by a 2d or a 4th, or a 6th (8th is the fundamental, does not add color).
>- With these rules, A C7sus2 add11 is equal to a C9sus4. So we add that we always prefer adding the lowest stacked third. Here the 11th is a 4th and the 2d is a ninth, it's a C9sus4.  

**d. Rule 4: simplification attempt**

    - The chords are build with stacked thirds + extensions that are not stacked thirds.
    - The fundamental and the fifth are constituent, so they are always considered here. The fifth can be omitted ; not the fundamental.
    - There cannot be an extension between two stacked thirds, so there get converted to stacked thirds.
    - The conversion goes from the lowest extension to the highest, until the rule is met.


**Examples applying rule 4**

1. C chord. If I add a D, it gets between C and E -> it's a 9th
2. C7 chord. If I add an A and a D, it gets between G and Bb, and between C and E -> it's a 9th and a 13th
7   
-> whether there is a 5th or not
3. C chord. If I add an A and a D, it's a 6th and a 9th
4. C7 no3 chord. If I add a D and an A, the D is a 2d and the A is a 13th.
5. C + D + F + A + Bb -> Not working

#### Conclusion
> [!note]
>
> Adding degrees over 6 makes it complicated to study alterations (9th degree of the chord VS 2d degree of the scale ; comparing a sus2 chord to a b9th chord...)
>
> The notion of 2d/9th is one of **playback**. The data structure, in order to be the most general possible, will only take the degrees of the scale, but implicitly a 2d, 4th, 6th will always be considered as a 9th, 11th, 13th for the time being.
>
> This means that a C9 chord is coded as `[0,1,2,4,6]` in terms of degrees.
>
> However, this also means the chord notes are not sorted in the right order, so ordering by semitone will always be just a rotation of the degree ordering.

```python
# Combining both representations : F scale
[
    [0,  1,  2,  3,  4,  5,  6  ],
    [5,  7,  9,  10, 0,  2,  4  ],
    [4,  5,  6,  0,  1,  2,  3  ], # These lines are a rotation of the two first lines by 4
    [0,  2,  4,  5,  7,  9,  10 ] 
]
# C9 no3
[
    [0,  1,  4,  6, ],
    [0,  2,  7,  10 ],
    [0,  1,  4,  6  ], # idem, but rotation index = 0 because the root is C
    [0,  2,  7,  10 ]
]
# F9 no3
[
    [0,  1,  4,  6, ],
    [5,  7,  0,  3  ],
    [4,  6,  0,  1  ], # rotation index = 2
    [0,  3,  5,  7  ]
]
```




### Combination ?
Perhaps the data structure should combine a way to access efficiently the chroma representation as well as the degree representation, ~~and the two types of degree representation.~~

The two types are just a reindexing.
- Standard order : Degree.
    - $j_0 = j \text{ t.q. } M_{0,j}=0: M' = (M)_{i, (j+j_0 \mod 3)}$
- Secondary order : Semitones



```python

C_scale_ = [
    [1,0,2,0,3,4,0,5,0,6,0,7],
    [0,1,2,3,4,5,6,7,8,9,10,11]
]

F_scale_ = [
    [5,0,6,0,7,1,0,2,0,3,4,0],
    [0,1,2,3,4,5,6,7,8,9,10,11]
]

C = [
    [1,0,0,0,3,0,0,5,0,0,0,0],
    [0,1,2,3,4,5,6,7,8,9,10,11]
]

```



## Analyser for a given chord sequence
This is also the specifying step.

- 2 types of metrics: intrisic (intrisic dissonance), relative (similarity). They can be combined : dissonance is the result of both for exemple and gives a relative dissonance. Cyclic position can be a result of CycleV and CycleIII and CycleII 

- These metrics can be used sequentially, to study their evolution through time with different windows of context. These sequential metrics are what will give the most important meta-parameters : they describe the **paths** of the chord progression. (example march of fifths of chromatic descent)

- The first analyser must have as an input :
    - A chord sequence
    - A number of metrics, intrisic or relative
- And give as an output :
    - A matrix of the metrics for the whole sequence

### To specify
- According to a given relative metric (example : similarity between two chords) how to format the data to provide statistics on several levels, for the path of the metric through time.
- Linked but different : how to define a tonal center and its evolution.

### First analyser
Only triads coming from 7-degree scales with limited metrics, in chroma representation
#### Metrics
Absolute:
    - dissonance, several definitions
    - position in the cycle of fifths
    - root(s) and fundamental(s) (no inversion yet - chroma cyclic rep)
    - number of notes(=3)
Relative:
    - Degree of a chord in a scale (if any)
        - Harder : closest scale degree to a given chord
    - Distance between chords (several definitions)
    - Distance between a chord and a scale

#### Difficulty
What data format for a path ?

#### Extensions
- Add sevenths and others
- 


## Generator from one principle
Applying the concepts to one principle.
- Cycle of fifths. Easy
- Degrees and alterations. Hard. Need to include the Degree object and alterations.

Main difficulty: find a good data format for what a chord progression is within a given metric, and the computation of parameters on the sequence (movement, attraction, etc). Define good meta-parameters to bias the generation.


### Adding principles
Generalising the analyser, and then the generator
- Which principles ?
- How are the principles intertwined ?
- How to transform them into useful meta-parameters ?



### Adding the iterative mode



