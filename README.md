# Viterbi
Implementation of the viterbi algorithm using the occasionally dishonest casino example from the Durbin book.
## Usage
```
python viterbi.py
```
## Example
Example from the durbin book.
``` console 
durbin
```
Result:
```
Rolls:     315116246446644245311321631164152133625144543631656626566666
Die:       FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLLLLL
Durbin:    FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLL
Viterbi:   FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLL
```
Posterior probabilities:
``` console 
posterior
```
Result:
```
Rolls:     315116246446644245311321631164152133625144543631656626566666
Die:       FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLLLLL
Posterior: FFFFFFFFFFFLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLL
Viterbi:   FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLL
```
Mark unsafe positions of the viterbi algorithm:
``` console 
marked
```
Result:
```
Rolls:     315116246446644245311321631164152133625144543631656626566666
Die:       FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLLLLL
Viterbi:   FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLL
Marked:    FFFFFFFFFFFXXFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFXXLLLLLLLLLL
Extended:  FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFXXXXLLLLLLLLL
```
