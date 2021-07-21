# Use HSL Solvers with CasADI installed via pip

Got the following error after following the "Obtaining HSL" instructions instructions with CasADI installed via pip on Fedora 30.
```
Exception of type: OPTION_INVALID in file "../../../../Ipopt/src/Algorithm/IpAlgBuilder.cpp" at line 321:
 Exception message: Selected linear solver MA57 not available.
Tried to obtain MA57 from shared library "libhsl.so", but the following error occured:
/lib/libhsl.so: undefined symbol: metis_nodend_
```
I ended up compiling the HSL code using [ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL), which is one of the options that Ipopt gives in its [documentation](https://coin-or.github.io/Ipopt/INSTALL.html), and then updating `$LD_LIBRARY_PATH` so it can be found at runtime.

First get `ThirdParty-HSl`.
```
git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
```
Unpack HSL codes.
```
gunzip coinhsl-x.y.z.tar.gz
tar xf coinhsl-x.y.z.tar
```
Rename `coinhsl-x.y.z` to `coinhsl` and move it into `ThirdParty-HSL`. Its new location should be `ThirdParty-HSL/coinhsl`. Then in `ThirdParty-HSL` run:
```
./configure
make
sudo make install
```
The shared library `libcoinhsl.so` will be located in `/usr/local/lib`. Since Ipopt looks for `libhsl.so` make a symbolic link from `libcoinhsl.so` to `libhsl.so`.
```
cd /usr/local/lib
ln -s libcoinhsl.so libhsl.so
```
Then specify the shared library path.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```
