"""Explicit static-library build; never invoke without native lane approval."""
from pathlib import Path
import argparse,os,subprocess
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
here=Path(__file__).resolve().parent
shader=(here/'field.h').read_text()+(here/'stage2.metal').read_text();assert ')OWNED_STAGE2"' not in shader
(a.output/'shader_source.h').write_text('static const char shader_source[]=R"OWNED_STAGE2('+shader+')OWNED_STAGE2";\n')
cmd=['/usr/bin/clang++','-std=c++17','-O2','-fobjc-arc','-fexceptions','-I',str(a.output),'-c',str(here/'owner.mm'),'-o',str(a.output/'owner.o')]
if os.environ.get('SDKROOT'):cmd[1:1]=['-isysroot',os.environ['SDKROOT']]
r=subprocess.run(cmd,capture_output=True);(a.output/'compile.stdout').write_bytes(r.stdout);(a.output/'compile.stderr').write_bytes(r.stderr)
if r.returncode:raise SystemExit(r.returncode)
r=subprocess.run(['/usr/bin/ar','rcs',str(a.output/'libakita_stage2_owned.a'),str(a.output/'owner.o')],capture_output=True);(a.output/'archive.stdout').write_bytes(r.stdout);(a.output/'archive.stderr').write_bytes(r.stderr)
if r.returncode:raise SystemExit(r.returncode)
cmd=['/usr/bin/clang++','-std=c++17','-O2',str(here/'fixture_client.cpp'),str(a.output/'libakita_stage2_owned.a'),'-framework','Metal','-framework','Foundation','-o',str(a.output/'owned-fixture')]
if os.environ.get('SDKROOT'):cmd[1:1]=['-isysroot',os.environ['SDKROOT']]
r=subprocess.run(cmd,capture_output=True);(a.output/'client.stdout').write_bytes(r.stdout);(a.output/'client.stderr').write_bytes(r.stderr);raise SystemExit(r.returncode)
