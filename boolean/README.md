# Boolean cuts

A 2 m x 1 m plate and a sphere 1 m in diameter, 0.5 m above it, each cut by
two cutters. The sphere is cut by a cylinder that goes through it and by a cone
opening upwards from its centre. The plate is cut by a tilted cone, which
leaves an elliptical hole, and by a sphere, which leaves a round one. A cutter
is a closed solid: cylinders and cones cut with their end caps.

`target - cutter` removes the volume of the cutter from the target, and `+`
groups the results into one model. Only the cut geometry is added to the
model, so the cutters are not shown. The script opens the interactive viewer
on the result.

There are no settings. Change the primitives in the script to change the
cuts.

```bash
python boolean.py
```
