# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Module 3.4 Stuff

### Plot
<img src="imgs/plot34.png" width="75%">

### Raw Data
#### Timing summary
Size: 64
* fast: 0.00545
* gpu: 0.01455

Size: 128
* fast: 0.01786
* gpu: 0.02563

Size: 256
* fast: 0.07524
* gpu: 0.05100

Size: 512
* fast: 0.49722
* gpu: 0.19878

Size: 1024
* fast: 8.69253
* gpu: 0.88500


## CPU Split Dataset (Hidden Layer 100, LR of 0.05)

Average time per epoch: 0.0799858 seconds

### Loss Data
Epoch  0  loss  8.158331075416443 correct 31\
Epoch  10  loss  7.437599609483676 correct 31\
Epoch  20  loss  5.266467804046692 correct 42\
Epoch  30  loss  6.127489741148496 correct 44\
Epoch  40  loss  6.32287763757689 correct 47\
Epoch  50  loss  3.5937361502574583 correct 50\
Epoch  60  loss  2.9464597536641843 correct 49\
Epoch  70  loss  2.3083101191960402 correct 49\
Epoch  80  loss  3.281094219092092 correct 49\
Epoch  90  loss  2.273324330804891 correct 47\
Epoch  100  loss  2.9036972015647353 correct 49\
Epoch  110  loss  1.7240296306482379 correct 49\
Epoch  120  loss  1.8539574134882364 correct 48\
Epoch  130  loss  2.5364299783846684 correct 49\
Epoch  140  loss  2.6421948598536362 correct 47\
Epoch  150  loss  2.552153819277846 correct 47\
Epoch  160  loss  1.109587130552987 correct 50\
Epoch  170  loss  0.8732575599501115 correct 48\
Epoch  180  loss  1.9998000400991935 correct 50

## CPU Xor Dataset (Hidden Layer 100, LR of 0.05)

Average time per epoch: 0.0780816 seconds

### Loss Data
Epoch  0  loss  6.963324767279923 correct 20\
Epoch  10  loss  5.45777000491824 correct 36\
Epoch  20  loss  4.87457880563027 correct 35\
Epoch  30  loss  4.558274762855238 correct 36\
Epoch  40  loss  4.014774153322469 correct 45\
Epoch  50  loss  3.7646734342333 correct 36\
Epoch  60  loss  6.13250661043877 correct 43\
Epoch  70  loss  4.21083551455361 correct 46\
Epoch  80  loss  4.251212141065572 correct 42\
Epoch  90  loss  3.803611009394337 correct 47\
Epoch  100  loss  3.463910071285496 correct 42\
Epoch  110  loss  2.5050982842469347 correct 44\
Epoch  120  loss  4.048149089347216 correct 46\
Epoch  130  loss  2.4252195167359742 correct 47\
Epoch  140  loss  2.860466436383708 correct 47\
Epoch  150  loss  1.9413327065280845 correct 47\
Epoch  160  loss  2.8012195311355077 correct 46\
Epoch  170  loss  2.174051028601034 correct 48\
Epoch  180  loss  2.7060529162380473 correct 46\
Epoch  190  loss  1.1248673094154509 correct 48\
Epoch  200  loss  2.2520491591163077 correct 45\
Epoch  210  loss  0.4388392911593749 correct 45\
Epoch  220  loss  1.4666663885412863 correct 48\
Epoch  230  loss  1.4136876189911727 correct 49\
Epoch  240  loss  1.6202146214801765 correct 49\
Epoch  250  loss  0.9333886271850071 correct 45\
Epoch  260  loss  0.21030483756959817 correct 46\
Epoch  270  loss  1.6515248837757193 correct 48\
Epoch  280  loss  1.1210577954713772 correct 49\
Epoch  290  loss  1.109957957347805 correct 48\
Epoch  300  loss  1.6284584880778346 correct 47\
Epoch  310  loss  0.35605587228822694 correct 47\
Epoch  320  loss  0.4553031094436542 correct 50\
Epoch  330  loss  0.3793793825857032 correct 45\
Epoch  340  loss  1.2802768109023361 correct 50\
Epoch  350  loss  0.480218282644271 correct 48\
Epoch  360  loss  0.40893864501606436 correct 46\
Epoch  370  loss  1.2853273805142622 correct 50

## CPU Simple Dataset (Hidden Layer 100, LR of 0.05)

Average time per epoch: 0.0847381 seconds

### Loss Data
Epoch  0  loss  4.386987428910761 correct 30\
Epoch  10  loss  1.537137494703264 correct 47\
Epoch  20  loss  1.3535009524845734 correct 49\
Epoch  30  loss  0.6428510757025829 correct 50\
Epoch  40  loss  1.6528048324982814 correct 47\
Epoch  50  loss  1.4331737038192442 correct 50\
Epoch  60  loss  0.9905884659905021 correct 49\
Epoch  70  loss  1.0522839560493344 correct 50

## GPU Split Dataset (Hidden Layer 100, LR of 0.05)

Average time per epoch: 2.6070413 seconds

### Loss Data
Epoch  0  loss  6.840350210119134 correct 31\
Epoch  10  loss  3.4110235532733677 correct 38\
Epoch  20  loss  4.408219400736454 correct 42\
Epoch  30  loss  2.776835696002391 correct 41\
Epoch  40  loss  3.964960111738579 correct 41\
Epoch  50  loss  3.0798404970991102 correct 48\
Epoch  60  loss  1.086752331486119 correct 44\
Epoch  70  loss  3.5820589825766858 correct 49\
Epoch  80  loss  1.6311496716079303 correct 49\
Epoch  90  loss  1.766665932212916 correct 49\
Epoch  100  loss  1.8572766479410705 correct 50\
Epoch  110  loss  2.681664579378218 correct 49\
Epoch  120  loss  1.426794110487073 correct 49\
Epoch  130  loss  0.9715168558852108 correct 49\
Epoch  140  loss  0.7424577748469872 correct 50\
Epoch  150  loss  2.005868650170111 correct 49\
Epoch  160  loss  0.45455410411631575 correct 50

## GPU Xor Dataset (Hidden Layer 100, LR of 0.05)

Average time per epoch: 2.8542119 seconds

### Loss Data
Epoch  0  loss  7.246203510350952 correct 32\
Epoch  10  loss  5.288182695729239 correct 44\
Epoch  20  loss  5.188813924795028 correct 46\
Epoch  30  loss  3.7688500384010872 correct 45\
Epoch  40  loss  2.0169916999487634 correct 47\
Epoch  50  loss  3.5352237882025035 correct 48\
Epoch  60  loss  4.6150442423830516 correct 45\
Epoch  70  loss  2.48171359779481 correct 47\
Epoch  80  loss  1.3809887259316747 correct 48\
Epoch  90  loss  2.5912771951306954 correct 47\
Epoch  100  loss  1.0814653863612058 correct 47\
Epoch  110  loss  4.673912731668265 correct 50\
Epoch  120  loss  1.4157393532728295 correct 48\
Epoch  130  loss  1.9795740107645676 correct 47\
Epoch  140  loss  1.4029202090628659 correct 49\
Epoch  150  loss  0.7845117264051655 correct 48\
Epoch  160  loss  0.7195255482093431 correct 48\
Epoch  170  loss  1.694457266390305 correct 50\
Epoch  180  loss  0.8100028192942718 correct 47\
Epoch  190  loss  0.6771936792814539 correct 47\
Epoch  200  loss  0.5224077903164924 correct 47\
Epoch  210  loss  2.029001271108087 correct 50

## GPU Simple Dataset (Hidden Layer 100, LR of 0.05)

Average time per epoch: 2.6921773 seconds

### Loss Data
Epoch  0  loss  5.547381384774088 correct 40\
Epoch  10  loss  2.34969288663491 correct 49\
Epoch  20  loss  1.2827078725508145 correct 50\
Epoch  30  loss  0.09780548182156365 correct 49\
Epoch  40  loss  1.5496472118858655 correct 50\
Epoch  50  loss  0.7052121010513221 correct 49\
Epoch  60  loss  0.5411754874697448 correct 49\
Epoch  70  loss  0.3644210895141787 correct 50