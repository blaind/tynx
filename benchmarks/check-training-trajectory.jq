def close($left; $right; $tolerance): (($left - $right) | abs) <= $tolerance;

.[0] as $left | .[1] as $right |
(($left | length) == ($right | length)) and
all(range(0; ($left | length));
  . as $case |
  $left[$case] as $a | $right[$case] as $b |
  ($a.case == $b.case) and
  ($a.model_sha256 == $b.model_sha256) and
  ($a.correctness.initial.trainable == $b.correctness.initial.trainable) and
  ($a.correctness.initial.frozen == $b.correctness.initial.frozen) and
  ($a.correctness.initial.parameter_sha256 == $b.correctness.initial.parameter_sha256) and
  (($a.correctness.trajectory | length) == ($b.correctness.trajectory | length)) and
  all(range(0; ($a.correctness.trajectory | length));
    . as $step |
    close(
      $a.correctness.trajectory[$step].state.loss;
      $b.correctness.trajectory[$step].state.loss;
      0.0001
    ) and
    ($a.correctness.trajectory[$step].state.gradients == $b.correctness.trajectory[$step].state.gradients) and
    ($a.correctness.trajectory[$step].state.parameter_sha256 == $b.correctness.trajectory[$step].state.parameter_sha256)
  )
)
