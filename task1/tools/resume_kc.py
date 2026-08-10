"""kc for a given resume epoch.

The curriculum factor lives in the env, not the checkpoint, so a resumed run
restarts it at kc_initial and silently undoes however far the curriculum had
got. Pass the printed value as KC0 when resuming.

    python tools/resume_kc.py 1500
"""
import sys

kc0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
kd = float(sys.argv[3]) if len(sys.argv) > 3 else 0.997
n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print("kc_%d = %.6f   (kc0=%g, kd=%g)" % (n, kc0 ** (kd ** n), kc0, kd))
