ޚ2
�5�5
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
Q
CheckNumerics
tensor"T
output"T"
Ttype:
2"
messagestring
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
n
LeakyReluGrad
	gradients"T
features"T
	backprops"T"
alphafloat%��L>"
Ttype0:
2
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
�
StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.13.12b'v1.13.0-rc2-5-g6612da8'��0
x
ppo_agent/ppo2_model/ObPlaceholder*
dtype0*
shape:*&
_output_shapes
:
�
Lppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
Jppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
valueB
 *����*
dtype0*
_output_shapes
: 
�
Jppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
Tppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/RandomUniformRandomUniformLppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/shape*
dtype0*
seed2*
seed�A*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
Jppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/subSubJppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/maxJppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
_output_shapes
: 
�
Jppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/mulMulTppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/RandomUniformJppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
Fppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniformAddJppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/mulJppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
+ppo_agent/ppo2_model/pi/conv_initial/kernel
VariableV2*
shape:*
shared_name *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
dtype0*
	container *&
_output_shapes
:
�
2ppo_agent/ppo2_model/pi/conv_initial/kernel/AssignAssign+ppo_agent/ppo2_model/pi/conv_initial/kernelFppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
0ppo_agent/ppo2_model/pi/conv_initial/kernel/readIdentity+ppo_agent/ppo2_model/pi/conv_initial/kernel*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
;ppo_agent/ppo2_model/pi/conv_initial/bias/Initializer/zerosConst*
dtype0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
valueB*    *
_output_shapes
:
�
)ppo_agent/ppo2_model/pi/conv_initial/bias
VariableV2*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
dtype0*
	container *
shape:*
shared_name *
_output_shapes
:
�
0ppo_agent/ppo2_model/pi/conv_initial/bias/AssignAssign)ppo_agent/ppo2_model/pi/conv_initial/bias;ppo_agent/ppo2_model/pi/conv_initial/bias/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
.ppo_agent/ppo2_model/pi/conv_initial/bias/readIdentity)ppo_agent/ppo2_model/pi/conv_initial/bias*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
2ppo_agent/ppo2_model/pi/conv_initial/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
�
+ppo_agent/ppo2_model/pi/conv_initial/Conv2DConv2Dppo_agent/ppo2_model/Ob0ppo_agent/ppo2_model/pi/conv_initial/kernel/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*&
_output_shapes
:
�
,ppo_agent/ppo2_model/pi/conv_initial/BiasAddBiasAdd+ppo_agent/ppo2_model/pi/conv_initial/Conv2D.ppo_agent/ppo2_model/pi/conv_initial/bias/read*
T0*
data_formatNHWC*&
_output_shapes
:
�
.ppo_agent/ppo2_model/pi/conv_initial/LeakyRelu	LeakyRelu,ppo_agent/ppo2_model/pi/conv_initial/BiasAdd*
T0*
alpha%��L>*&
_output_shapes
:
�
Fppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/shapeConst*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
Dppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/minConst*
dtype0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
valueB
 *�{�*
_output_shapes
: 
�
Dppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/maxConst*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
valueB
 *�{�=*
dtype0*
_output_shapes
: 
�
Nppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformFppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/shape*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
dtype0*
seed2*
seed�A*&
_output_shapes
:
�
Dppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/subSubDppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/maxDppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/min*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
_output_shapes
: 
�
Dppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/mulMulNppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/RandomUniformDppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/sub*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
@ppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniformAddDppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/mulDppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform/min*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
%ppo_agent/ppo2_model/pi/conv_0/kernel
VariableV2*
shared_name *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
dtype0*
	container *
shape:*&
_output_shapes
:
�
,ppo_agent/ppo2_model/pi/conv_0/kernel/AssignAssign%ppo_agent/ppo2_model/pi/conv_0/kernel@ppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_0/kernel/readIdentity%ppo_agent/ppo2_model/pi/conv_0/kernel*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
5ppo_agent/ppo2_model/pi/conv_0/bias/Initializer/zerosConst*
dtype0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
valueB*    *
_output_shapes
:
�
#ppo_agent/ppo2_model/pi/conv_0/bias
VariableV2*
shared_name *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
dtype0*
	container *
shape:*
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_0/bias/AssignAssign#ppo_agent/ppo2_model/pi/conv_0/bias5ppo_agent/ppo2_model/pi/conv_0/bias/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
(ppo_agent/ppo2_model/pi/conv_0/bias/readIdentity#ppo_agent/ppo2_model/pi/conv_0/bias*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
}
,ppo_agent/ppo2_model/pi/conv_0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
%ppo_agent/ppo2_model/pi/conv_0/Conv2DConv2D.ppo_agent/ppo2_model/pi/conv_initial/LeakyRelu*ppo_agent/ppo2_model/pi/conv_0/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:
�
&ppo_agent/ppo2_model/pi/conv_0/BiasAddBiasAdd%ppo_agent/ppo2_model/pi/conv_0/Conv2D(ppo_agent/ppo2_model/pi/conv_0/bias/read*
data_formatNHWC*
T0*&
_output_shapes
:
�
(ppo_agent/ppo2_model/pi/conv_0/LeakyRelu	LeakyRelu&ppo_agent/ppo2_model/pi/conv_0/BiasAdd*
T0*
alpha%��L>*&
_output_shapes
:
�
Fppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/shapeConst*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
Dppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/minConst*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
valueB
 *�{�*
dtype0*
_output_shapes
: 
�
Dppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/maxConst*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
valueB
 *�{�=*
dtype0*
_output_shapes
: 
�
Nppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformFppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/shape*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
dtype0*
seed2(*
seed�A*&
_output_shapes
:
�
Dppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/subSubDppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/maxDppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/min*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
_output_shapes
: 
�
Dppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/mulMulNppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/RandomUniformDppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/sub*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
@ppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniformAddDppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/mulDppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform/min*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
%ppo_agent/ppo2_model/pi/conv_1/kernel
VariableV2*
dtype0*
	container *
shape:*
shared_name *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
,ppo_agent/ppo2_model/pi/conv_1/kernel/AssignAssign%ppo_agent/ppo2_model/pi/conv_1/kernel@ppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_1/kernel/readIdentity%ppo_agent/ppo2_model/pi/conv_1/kernel*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
5ppo_agent/ppo2_model/pi/conv_1/bias/Initializer/zerosConst*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
#ppo_agent/ppo2_model/pi/conv_1/bias
VariableV2*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
dtype0*
	container *
shape:*
shared_name *
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_1/bias/AssignAssign#ppo_agent/ppo2_model/pi/conv_1/bias5ppo_agent/ppo2_model/pi/conv_1/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
(ppo_agent/ppo2_model/pi/conv_1/bias/readIdentity#ppo_agent/ppo2_model/pi/conv_1/bias*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
}
,ppo_agent/ppo2_model/pi/conv_1/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
�
%ppo_agent/ppo2_model/pi/conv_1/Conv2DConv2D(ppo_agent/ppo2_model/pi/conv_0/LeakyRelu*ppo_agent/ppo2_model/pi/conv_1/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
&ppo_agent/ppo2_model/pi/conv_1/BiasAddBiasAdd%ppo_agent/ppo2_model/pi/conv_1/Conv2D(ppo_agent/ppo2_model/pi/conv_1/bias/read*
data_formatNHWC*
T0*&
_output_shapes
:
�
(ppo_agent/ppo2_model/pi/conv_1/LeakyRelu	LeakyRelu&ppo_agent/ppo2_model/pi/conv_1/BiasAdd*
T0*
alpha%��L>*&
_output_shapes
:
~
-ppo_agent/ppo2_model/pi/flatten/Reshape/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
�
'ppo_agent/ppo2_model/pi/flatten/ReshapeReshape(ppo_agent/ppo2_model/pi/conv_1/LeakyRelu-ppo_agent/ppo2_model/pi/flatten/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	�
�
Eppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/shapeConst*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
valueB"�  @   *
dtype0*
_output_shapes
:
�
Cppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/minConst*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
valueB
 *PEݽ*
dtype0*
_output_shapes
: 
�
Cppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
valueB
 *PE�=*
_output_shapes
: 
�
Mppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniformEppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/shape*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
dtype0*
seed2<*
seed�A*
_output_shapes
:	�@
�
Cppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/subSubCppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/maxCppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
: 
�
Cppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/mulMulMppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/RandomUniformCppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/sub*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
?ppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniformAddCppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/mulCppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
$ppo_agent/ppo2_model/pi/dense/kernel
VariableV2*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
dtype0*
	container *
shape:	�@*
shared_name *
_output_shapes
:	�@
�
+ppo_agent/ppo2_model/pi/dense/kernel/AssignAssign$ppo_agent/ppo2_model/pi/dense/kernel?ppo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
)ppo_agent/ppo2_model/pi/dense/kernel/readIdentity$ppo_agent/ppo2_model/pi/dense/kernel*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
4ppo_agent/ppo2_model/pi/dense/bias/Initializer/zerosConst*
dtype0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
valueB@*    *
_output_shapes
:@
�
"ppo_agent/ppo2_model/pi/dense/bias
VariableV2*
dtype0*
	container *
shape:@*
shared_name *5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
)ppo_agent/ppo2_model/pi/dense/bias/AssignAssign"ppo_agent/ppo2_model/pi/dense/bias4ppo_agent/ppo2_model/pi/dense/bias/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
'ppo_agent/ppo2_model/pi/dense/bias/readIdentity"ppo_agent/ppo2_model/pi/dense/bias*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
$ppo_agent/ppo2_model/pi/dense/MatMulMatMul'ppo_agent/ppo2_model/pi/flatten/Reshape)ppo_agent/ppo2_model/pi/dense/kernel/read*
T0*
transpose_a( *
transpose_b( *
_output_shapes

:@
�
%ppo_agent/ppo2_model/pi/dense/BiasAddBiasAdd$ppo_agent/ppo2_model/pi/dense/MatMul'ppo_agent/ppo2_model/pi/dense/bias/read*
T0*
data_formatNHWC*
_output_shapes

:@
�
'ppo_agent/ppo2_model/pi/dense/LeakyRelu	LeakyRelu%ppo_agent/ppo2_model/pi/dense/BiasAdd*
alpha%��L>*
T0*
_output_shapes

:@
�
Gppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/shapeConst*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
�
Eppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/minConst*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
Eppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/maxConst*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
valueB
 *׳]>*
dtype0*
_output_shapes
: 
�
Oppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformGppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
seed2M*
seed�A*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
Eppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/subSubEppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/maxEppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes
: 
�
Eppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/mulMulOppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformEppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/sub*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
Appo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniformAddEppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/mulEppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
&ppo_agent/ppo2_model/pi/dense_1/kernel
VariableV2*
dtype0*
	container *
shape
:@@*
shared_name *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
-ppo_agent/ppo2_model/pi/dense_1/kernel/AssignAssign&ppo_agent/ppo2_model/pi/dense_1/kernelAppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
+ppo_agent/ppo2_model/pi/dense_1/kernel/readIdentity&ppo_agent/ppo2_model/pi/dense_1/kernel*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
6ppo_agent/ppo2_model/pi/dense_1/bias/Initializer/zerosConst*
dtype0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
valueB@*    *
_output_shapes
:@
�
$ppo_agent/ppo2_model/pi/dense_1/bias
VariableV2*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
dtype0*
	container *
shape:@*
_output_shapes
:@
�
+ppo_agent/ppo2_model/pi/dense_1/bias/AssignAssign$ppo_agent/ppo2_model/pi/dense_1/bias6ppo_agent/ppo2_model/pi/dense_1/bias/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
)ppo_agent/ppo2_model/pi/dense_1/bias/readIdentity$ppo_agent/ppo2_model/pi/dense_1/bias*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
&ppo_agent/ppo2_model/pi/dense_1/MatMulMatMul'ppo_agent/ppo2_model/pi/dense/LeakyRelu+ppo_agent/ppo2_model/pi/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b( *
_output_shapes

:@
�
'ppo_agent/ppo2_model/pi/dense_1/BiasAddBiasAdd&ppo_agent/ppo2_model/pi/dense_1/MatMul)ppo_agent/ppo2_model/pi/dense_1/bias/read*
T0*
data_formatNHWC*
_output_shapes

:@
�
)ppo_agent/ppo2_model/pi/dense_1/LeakyRelu	LeakyRelu'ppo_agent/ppo2_model/pi/dense_1/BiasAdd*
T0*
alpha%��L>*
_output_shapes

:@
�
Gppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
�
Eppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/minConst*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
Eppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/maxConst*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
valueB
 *׳]>*
dtype0*
_output_shapes
: 
�
Oppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniformGppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/shape*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
dtype0*
seed2^*
seed�A*
_output_shapes

:@@
�
Eppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/subSubEppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/maxEppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes
: 
�
Eppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/mulMulOppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformEppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
Appo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniformAddEppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/mulEppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
&ppo_agent/ppo2_model/pi/dense_2/kernel
VariableV2*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
dtype0*
	container *
shape
:@@*
shared_name *
_output_shapes

:@@
�
-ppo_agent/ppo2_model/pi/dense_2/kernel/AssignAssign&ppo_agent/ppo2_model/pi/dense_2/kernelAppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
+ppo_agent/ppo2_model/pi/dense_2/kernel/readIdentity&ppo_agent/ppo2_model/pi/dense_2/kernel*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
6ppo_agent/ppo2_model/pi/dense_2/bias/Initializer/zerosConst*
dtype0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
valueB@*    *
_output_shapes
:@
�
$ppo_agent/ppo2_model/pi/dense_2/bias
VariableV2*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
dtype0*
	container *
shape:@*
_output_shapes
:@
�
+ppo_agent/ppo2_model/pi/dense_2/bias/AssignAssign$ppo_agent/ppo2_model/pi/dense_2/bias6ppo_agent/ppo2_model/pi/dense_2/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
)ppo_agent/ppo2_model/pi/dense_2/bias/readIdentity$ppo_agent/ppo2_model/pi/dense_2/bias*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
&ppo_agent/ppo2_model/pi/dense_2/MatMulMatMul)ppo_agent/ppo2_model/pi/dense_1/LeakyRelu+ppo_agent/ppo2_model/pi/dense_2/kernel/read*
transpose_a( *
transpose_b( *
T0*
_output_shapes

:@
�
'ppo_agent/ppo2_model/pi/dense_2/BiasAddBiasAdd&ppo_agent/ppo2_model/pi/dense_2/MatMul)ppo_agent/ppo2_model/pi/dense_2/bias/read*
data_formatNHWC*
T0*
_output_shapes

:@
�
)ppo_agent/ppo2_model/pi/dense_2/LeakyRelu	LeakyRelu'ppo_agent/ppo2_model/pi/dense_2/BiasAdd*
T0*
alpha%��L>*
_output_shapes

:@
]
ppo_agent/ppo2_model/ConstConst*
dtype0*
valueB *
_output_shapes
: 
{
*ppo_agent/ppo2_model/flatten/Reshape/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
�
$ppo_agent/ppo2_model/flatten/ReshapeReshape)ppo_agent/ppo2_model/pi/dense_2/LeakyRelu*ppo_agent/ppo2_model/flatten/Reshape/shape*
T0*
Tshape0*
_output_shapes

:@
}
,ppo_agent/ppo2_model/flatten_1/Reshape/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
�
&ppo_agent/ppo2_model/flatten_1/ReshapeReshape)ppo_agent/ppo2_model/pi/dense_2/LeakyRelu,ppo_agent/ppo2_model/flatten_1/Reshape/shape*
T0*
Tshape0*
_output_shapes

:@
�
3ppo_agent/ppo2_model/pi/w/Initializer/initial_valueConst*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*�
value�B�@"���8�WԹD��hK�:oON9x�48�|5�+:���n�P�Z���봺q:���:=Է��`�9���9���ke:�!��Nl�:�2[;�bv:	�4�h9�\���۹���f�L:B׻:\!�9ĕx�풖��u�Y�: �!�:��Y�~4d;L$[��xa�(��9K�;=�Ϻ3H۹�'�,�M������<;A��:i�۸o��:�����7y:׺�weP��׎��ͧ:c��85:���:��j;;����4z��ԣ:*R�t;j�:ܯ�:��:�:Q(�:CO��y�7\t�A&�O����z�e;&�J8�����:�h�:���:�.;���:���_YมI��K����UY���9�º,�ڹ��L�q���}:��:�7Һ�D�:O�9֤�:n�|x ;�����1:�t:��09��3��4�:�L��S�7V�K��E:=��9�Kt�DC���;}�Ǻ��9g�ź��
:v�*�:�W�9�SQ��;m��:�	��Ӫ�u��9Q/:dvP:Ļ<��j�0��8�L�9�����%�9�M�:?��:��y:�'|�
�F��ź	��9o˴:M=�:�N�9�&u��2�5�78+�9}�;P�i�Xa�8��8��d�:��\:�o�e84J;:(�9�&:<�o��119E��o�9�7B:�J/;K�J:�9͹6�}�������9���:|�.:��O;G:����ǜ��9�r:�:�A�:�Qv�R�_�26�:sݱ�N�����_�L�ͺ����0����}���8:���:���:�ֹa�:�4I���[�K*	��J::;L:y�9Bd:Ic:N洹�ʺD��a󈺽X�:ǯ�����~�,:d�
9oMúf��9�!��,�θ���4�9���b��33�:��O���9!��㷍:��Y����	�ﱍ�h1�:ˮ�9�ɽ7�C�!|�� ���]�]ܺ!�':�qѺ+j��T��d�2��U����:6F:��[�����:7ڔ�g��������:/	��c_�Z�:��+����hJ�:�X:�=�����/�9 	Q�� !�ߑI;�N޺��1��-���U�9�����պ��9@����B�-!�9��	����n:Oq����9M�:�����J���_T;;ۅA8y�9���:뒱:P�\�}��7{��	:���:l�Q�Y��T!�:(��:�=g��3#9�y��}
:-.�:�-;Y�͹�:�8�U�:bB:J�-9�t:�mߺ�	-:i�M:�U�7go���7ĺ!У:�$�z�8�8�6;�(+9d?�@Z�:</�x$��;G�:$"�9�s:�I̺&�:�R���*��x;y����-:�����j���89	�����:7E$9Q��m��:U����Z:Z�k��7:)�����2:��M:�.n:���9��3;�I��8�V9F���ƹ�;����*9�\=�W��:�K�N��9ZYй��:��(�teR:�;H�Q�:�	:]Y������9M��:*
dtype0*
_output_shapes

:@
�
ppo_agent/ppo2_model/pi/w
VariableV2*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
dtype0*
	container *
shape
:@*
shared_name *
_output_shapes

:@
�
 ppo_agent/ppo2_model/pi/w/AssignAssignppo_agent/ppo2_model/pi/w3ppo_agent/ppo2_model/pi/w/Initializer/initial_value*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
ppo_agent/ppo2_model/pi/w/readIdentityppo_agent/ppo2_model/pi/w*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
+ppo_agent/ppo2_model/pi/b/Initializer/ConstConst*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
valueB*    *
dtype0*
_output_shapes
:
�
ppo_agent/ppo2_model/pi/b
VariableV2*
dtype0*
	container *
shape:*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
 ppo_agent/ppo2_model/pi/b/AssignAssignppo_agent/ppo2_model/pi/b+ppo_agent/ppo2_model/pi/b/Initializer/Const*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
ppo_agent/ppo2_model/pi/b/readIdentityppo_agent/ppo2_model/pi/b*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
 ppo_agent/ppo2_model/pi_1/MatMulMatMul&ppo_agent/ppo2_model/flatten_1/Reshapeppo_agent/ppo2_model/pi/w/read*
T0*
transpose_a( *
transpose_b( *
_output_shapes

:
�
ppo_agent/ppo2_model/pi_1/addAdd ppo_agent/ppo2_model/pi_1/MatMulppo_agent/ppo2_model/pi/b/read*
T0*
_output_shapes

:
o
ppo_agent/ppo2_model/SoftmaxSoftmaxppo_agent/ppo2_model/pi_1/add*
T0*
_output_shapes

:
t
!ppo_agent/ppo2_model/action_probsIdentityppo_agent/ppo2_model/Softmax*
T0*
_output_shapes

:
k
ppo_agent/ppo2_model/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
l
'ppo_agent/ppo2_model/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
l
'ppo_agent/ppo2_model/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
1ppo_agent/ppo2_model/random_uniform/RandomUniformRandomUniformppo_agent/ppo2_model/Shape*
seed�A*
T0*
dtype0*
seed2�*
_output_shapes

:
�
'ppo_agent/ppo2_model/random_uniform/subSub'ppo_agent/ppo2_model/random_uniform/max'ppo_agent/ppo2_model/random_uniform/min*
T0*
_output_shapes
: 
�
'ppo_agent/ppo2_model/random_uniform/mulMul1ppo_agent/ppo2_model/random_uniform/RandomUniform'ppo_agent/ppo2_model/random_uniform/sub*
T0*
_output_shapes

:
�
#ppo_agent/ppo2_model/random_uniformAdd'ppo_agent/ppo2_model/random_uniform/mul'ppo_agent/ppo2_model/random_uniform/min*
T0*
_output_shapes

:
m
ppo_agent/ppo2_model/LogLog#ppo_agent/ppo2_model/random_uniform*
T0*
_output_shapes

:
b
ppo_agent/ppo2_model/NegNegppo_agent/ppo2_model/Log*
T0*
_output_shapes

:
d
ppo_agent/ppo2_model/Log_1Logppo_agent/ppo2_model/Neg*
T0*
_output_shapes

:
�
ppo_agent/ppo2_model/subSubppo_agent/ppo2_model/pi_1/addppo_agent/ppo2_model/Log_1*
T0*
_output_shapes

:
p
%ppo_agent/ppo2_model/ArgMax/dimensionConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
ppo_agent/ppo2_model/ArgMaxArgMaxppo_agent/ppo2_model/sub%ppo_agent/ppo2_model/ArgMax/dimension*

Tidx0*
T0*
output_type0	*
_output_shapes
:
i
ppo_agent/ppo2_model/actionIdentityppo_agent/ppo2_model/ArgMax*
T0	*
_output_shapes
:
j
%ppo_agent/ppo2_model/one_hot/on_valueConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
k
&ppo_agent/ppo2_model/one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
d
"ppo_agent/ppo2_model/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
�
ppo_agent/ppo2_model/one_hotOneHotppo_agent/ppo2_model/action"ppo_agent/ppo2_model/one_hot/depth%ppo_agent/ppo2_model/one_hot/on_value&ppo_agent/ppo2_model/one_hot/off_value*
T0*
TI0	*
axis���������*
_output_shapes

:
}
;ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:

=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Shape_1Const*
dtype0*
valueB"      *
_output_shapes
:
~
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
:ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/SubSub=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Rank_1<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
�
Bppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice/beginPack:ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N*
_output_shapes
:
�
Appo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/SliceSlice>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Shape_1Bppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice/beginAppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice/size*
T0*
Index0*
_output_shapes
:
�
Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
�
Bppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concatConcatV2Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat/values_0<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/SliceBppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat/axis*
T0*
N*

Tidx0*
_output_shapes
:
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/ReshapeReshapeppo_agent/ppo2_model/pi_1/add=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*
_output_shapes

:

=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Shape_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_1Sub=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Rank_2>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
�
Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1/beginPack<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_1*
T0*

axis *
N*
_output_shapes
:
�
Cppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1Slice>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Shape_2Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1/beginCppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1/size*
T0*
Index0*
_output_shapes
:
�
Hppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
�
Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1ConcatV2Hppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1/values_0>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_1Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Reshape_1Reshapeppo_agent/ppo2_model/one_hot?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*
_output_shapes

:
�
6ppo_agent/ppo2_model/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Reshape@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Reshape_1*
T0*$
_output_shapes
::
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_2/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_2Sub;ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Rank>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
�
Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
Cppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2/sizePack<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2Slice<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/ShapeDppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2/beginCppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2/size*
T0*
Index0*
_output_shapes
:
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Reshape_2Reshape6ppo_agent/ppo2_model/softmax_cross_entropy_with_logits>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*
_output_shapes
:
�
3ppo_agent/ppo2_model/vf/w/Initializer/initial_valueConst*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*�
value�B�@"��<!>�߈�Z0]��<�s��q�F]=��8��y����=F���89>�Ig<�=s�]n�<Z�U=�����N�-q��pr>n��>��>I9<Wy=��>?c�=|O%�{	e>�6X���W�~gZ>��"=N���j��)w<���=��!>e�E�1=���S�>p>��S��O��`�<:dZ<�4 �sv�	i]�f<�=�Խ����Z��=Z',=*Js>]���Wt>U����P���S��0>���=K�=�B=*
dtype0*
_output_shapes

:@
�
ppo_agent/ppo2_model/vf/w
VariableV2*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
dtype0*
	container *
shape
:@*
_output_shapes

:@
�
 ppo_agent/ppo2_model/vf/w/AssignAssignppo_agent/ppo2_model/vf/w3ppo_agent/ppo2_model/vf/w/Initializer/initial_value*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
ppo_agent/ppo2_model/vf/w/readIdentityppo_agent/ppo2_model/vf/w*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
+ppo_agent/ppo2_model/vf/b/Initializer/ConstConst*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
valueB*    *
dtype0*
_output_shapes
:
�
ppo_agent/ppo2_model/vf/b
VariableV2*
shape:*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
dtype0*
	container *
_output_shapes
:
�
 ppo_agent/ppo2_model/vf/b/AssignAssignppo_agent/ppo2_model/vf/b+ppo_agent/ppo2_model/vf/b/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
ppo_agent/ppo2_model/vf/b/readIdentityppo_agent/ppo2_model/vf/b*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
ppo_agent/ppo2_model/vf/MatMulMatMul$ppo_agent/ppo2_model/flatten/Reshapeppo_agent/ppo2_model/vf/w/read*
T0*
transpose_a( *
transpose_b( *
_output_shapes

:
�
ppo_agent/ppo2_model/vf/addAddppo_agent/ppo2_model/vf/MatMulppo_agent/ppo2_model/vf/b/read*
T0*
_output_shapes

:
y
(ppo_agent/ppo2_model/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
{
*ppo_agent/ppo2_model/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
{
*ppo_agent/ppo2_model/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
"ppo_agent/ppo2_model/strided_sliceStridedSliceppo_agent/ppo2_model/vf/add(ppo_agent/ppo2_model/strided_slice/stack*ppo_agent/ppo2_model/strided_slice/stack_1*ppo_agent/ppo2_model/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
o
ppo_agent/ppo2_model/valueIdentity"ppo_agent/ppo2_model/strided_slice*
T0*
_output_shapes
:
|
ppo_agent/ppo2_model/Ob_1Placeholder*
dtype0*
shape:�*'
_output_shapes
:�
�
4ppo_agent/ppo2_model/pi_2/conv_initial/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
-ppo_agent/ppo2_model/pi_2/conv_initial/Conv2DConv2Dppo_agent/ppo2_model/Ob_10ppo_agent/ppo2_model/pi/conv_initial/kernel/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*'
_output_shapes
:�
�
.ppo_agent/ppo2_model/pi_2/conv_initial/BiasAddBiasAdd-ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D.ppo_agent/ppo2_model/pi/conv_initial/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:�
�
0ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu	LeakyRelu.ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd*
alpha%��L>*
T0*'
_output_shapes
:�

.ppo_agent/ppo2_model/pi_2/conv_0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
'ppo_agent/ppo2_model/pi_2/conv_0/Conv2DConv2D0ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu*ppo_agent/ppo2_model/pi/conv_0/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�
�
(ppo_agent/ppo2_model/pi_2/conv_0/BiasAddBiasAdd'ppo_agent/ppo2_model/pi_2/conv_0/Conv2D(ppo_agent/ppo2_model/pi/conv_0/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:�
�
*ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu	LeakyRelu(ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd*
T0*
alpha%��L>*'
_output_shapes
:�

.ppo_agent/ppo2_model/pi_2/conv_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
�
'ppo_agent/ppo2_model/pi_2/conv_1/Conv2DConv2D*ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu*ppo_agent/ppo2_model/pi/conv_1/kernel/read*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*'
_output_shapes
:�
�
(ppo_agent/ppo2_model/pi_2/conv_1/BiasAddBiasAdd'ppo_agent/ppo2_model/pi_2/conv_1/Conv2D(ppo_agent/ppo2_model/pi/conv_1/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:�
�
*ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu	LeakyRelu(ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd*
alpha%��L>*
T0*'
_output_shapes
:�
�
/ppo_agent/ppo2_model/pi_2/flatten/Reshape/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
�
)ppo_agent/ppo2_model/pi_2/flatten/ReshapeReshape*ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu/ppo_agent/ppo2_model/pi_2/flatten/Reshape/shape*
T0*
Tshape0* 
_output_shapes
:
��
�
&ppo_agent/ppo2_model/pi_2/dense/MatMulMatMul)ppo_agent/ppo2_model/pi_2/flatten/Reshape)ppo_agent/ppo2_model/pi/dense/kernel/read*
T0*
transpose_a( *
transpose_b( *
_output_shapes
:	�@
�
'ppo_agent/ppo2_model/pi_2/dense/BiasAddBiasAdd&ppo_agent/ppo2_model/pi_2/dense/MatMul'ppo_agent/ppo2_model/pi/dense/bias/read*
T0*
data_formatNHWC*
_output_shapes
:	�@
�
)ppo_agent/ppo2_model/pi_2/dense/LeakyRelu	LeakyRelu'ppo_agent/ppo2_model/pi_2/dense/BiasAdd*
T0*
alpha%��L>*
_output_shapes
:	�@
�
(ppo_agent/ppo2_model/pi_2/dense_1/MatMulMatMul)ppo_agent/ppo2_model/pi_2/dense/LeakyRelu+ppo_agent/ppo2_model/pi/dense_1/kernel/read*
transpose_a( *
transpose_b( *
T0*
_output_shapes
:	�@
�
)ppo_agent/ppo2_model/pi_2/dense_1/BiasAddBiasAdd(ppo_agent/ppo2_model/pi_2/dense_1/MatMul)ppo_agent/ppo2_model/pi/dense_1/bias/read*
T0*
data_formatNHWC*
_output_shapes
:	�@
�
+ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu	LeakyRelu)ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd*
T0*
alpha%��L>*
_output_shapes
:	�@
�
(ppo_agent/ppo2_model/pi_2/dense_2/MatMulMatMul+ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu+ppo_agent/ppo2_model/pi/dense_2/kernel/read*
T0*
transpose_a( *
transpose_b( *
_output_shapes
:	�@
�
)ppo_agent/ppo2_model/pi_2/dense_2/BiasAddBiasAdd(ppo_agent/ppo2_model/pi_2/dense_2/MatMul)ppo_agent/ppo2_model/pi/dense_2/bias/read*
data_formatNHWC*
T0*
_output_shapes
:	�@
�
+ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu	LeakyRelu)ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd*
T0*
alpha%��L>*
_output_shapes
:	�@
_
ppo_agent/ppo2_model/Const_1Const*
dtype0*
valueB *
_output_shapes
: 
}
,ppo_agent/ppo2_model/flatten_2/Reshape/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
�
&ppo_agent/ppo2_model/flatten_2/ReshapeReshape+ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu,ppo_agent/ppo2_model/flatten_2/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	�@
}
,ppo_agent/ppo2_model/flatten_3/Reshape/shapeConst*
dtype0*
valueB"   ����*
_output_shapes
:
�
&ppo_agent/ppo2_model/flatten_3/ReshapeReshape+ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu,ppo_agent/ppo2_model/flatten_3/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	�@
�
 ppo_agent/ppo2_model/pi_3/MatMulMatMul&ppo_agent/ppo2_model/flatten_3/Reshapeppo_agent/ppo2_model/pi/w/read*
T0*
transpose_a( *
transpose_b( *
_output_shapes
:	�
�
ppo_agent/ppo2_model/pi_3/addAdd ppo_agent/ppo2_model/pi_3/MatMulppo_agent/ppo2_model/pi/b/read*
T0*
_output_shapes
:	�
r
ppo_agent/ppo2_model/Softmax_1Softmaxppo_agent/ppo2_model/pi_3/add*
T0*
_output_shapes
:	�
y
#ppo_agent/ppo2_model/action_probs_1Identityppo_agent/ppo2_model/Softmax_1*
T0*
_output_shapes
:	�
m
ppo_agent/ppo2_model/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
n
)ppo_agent/ppo2_model/random_uniform_1/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n
)ppo_agent/ppo2_model/random_uniform_1/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
3ppo_agent/ppo2_model/random_uniform_1/RandomUniformRandomUniformppo_agent/ppo2_model/Shape_1*
T0*
dtype0*
seed2�*
seed�A*
_output_shapes
:	�
�
)ppo_agent/ppo2_model/random_uniform_1/subSub)ppo_agent/ppo2_model/random_uniform_1/max)ppo_agent/ppo2_model/random_uniform_1/min*
T0*
_output_shapes
: 
�
)ppo_agent/ppo2_model/random_uniform_1/mulMul3ppo_agent/ppo2_model/random_uniform_1/RandomUniform)ppo_agent/ppo2_model/random_uniform_1/sub*
T0*
_output_shapes
:	�
�
%ppo_agent/ppo2_model/random_uniform_1Add)ppo_agent/ppo2_model/random_uniform_1/mul)ppo_agent/ppo2_model/random_uniform_1/min*
T0*
_output_shapes
:	�
r
ppo_agent/ppo2_model/Log_2Log%ppo_agent/ppo2_model/random_uniform_1*
T0*
_output_shapes
:	�
g
ppo_agent/ppo2_model/Neg_1Negppo_agent/ppo2_model/Log_2*
T0*
_output_shapes
:	�
g
ppo_agent/ppo2_model/Log_3Logppo_agent/ppo2_model/Neg_1*
T0*
_output_shapes
:	�
�
ppo_agent/ppo2_model/sub_1Subppo_agent/ppo2_model/pi_3/addppo_agent/ppo2_model/Log_3*
T0*
_output_shapes
:	�
r
'ppo_agent/ppo2_model/ArgMax_1/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
ppo_agent/ppo2_model/ArgMax_1ArgMaxppo_agent/ppo2_model/sub_1'ppo_agent/ppo2_model/ArgMax_1/dimension*
output_type0	*

Tidx0*
T0*
_output_shapes	
:�
n
ppo_agent/ppo2_model/action_1Identityppo_agent/ppo2_model/ArgMax_1*
T0	*
_output_shapes	
:�
l
'ppo_agent/ppo2_model/one_hot_1/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
m
(ppo_agent/ppo2_model/one_hot_1/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
f
$ppo_agent/ppo2_model/one_hot_1/depthConst*
value	B :*
dtype0*
_output_shapes
: 
�
ppo_agent/ppo2_model/one_hot_1OneHotppo_agent/ppo2_model/action_1$ppo_agent/ppo2_model/one_hot_1/depth'ppo_agent/ppo2_model/one_hot_1/on_value(ppo_agent/ppo2_model/one_hot_1/off_value*
T0*
TI0	*
axis���������*
_output_shapes
:	�

=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/SubSub?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Rank_1>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub/y*
T0*
_output_shapes
: 
�
Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice/beginPack<ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub*
N*
T0*

axis *
_output_shapes
:
�
Cppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/SliceSlice@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Shape_1Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice/beginCppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice/size*
T0*
Index0*
_output_shapes
:
�
Hppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
�
Dppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concatConcatV2Hppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat/values_0>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/SliceDppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/ReshapeReshapeppo_agent/ppo2_model/pi_3/add?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat*
T0*
Tshape0*
_output_shapes
:	�
�
?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Shape_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_1/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_1Sub?ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Rank_2@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_1/y*
T0*
_output_shapes
: 
�
Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1/beginPack>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_1*
T0*

axis *
N*
_output_shapes
:
�
Eppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1Slice@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Shape_2Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1/beginEppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1/size*
T0*
Index0*
_output_shapes
:
�
Jppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
�
Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Appo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1ConcatV2Jppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1/values_0@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_1Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1/axis*
T0*
N*

Tidx0*
_output_shapes
:
�
Bppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Reshape_1Reshapeppo_agent/ppo2_model/one_hot_1Appo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/concat_1*
T0*
Tshape0*
_output_shapes
:	�
�
8ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1SoftmaxCrossEntropyWithLogits@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/ReshapeBppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Reshape_1*
T0*&
_output_shapes
:�:	�
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_2Sub=ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Rank@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_2/y*
T0*
_output_shapes
: 
�
Fppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2/beginConst*
dtype0*
valueB: *
_output_shapes
:
�
Eppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2/sizePack>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2Slice>ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/ShapeFppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2/beginEppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2/size*
T0*
Index0*
_output_shapes
:
�
Bppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Reshape_2Reshape8ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1@ppo_agent/ppo2_model/softmax_cross_entropy_with_logits_1/Slice_2*
T0*
Tshape0*
_output_shapes	
:�
�
 ppo_agent/ppo2_model/vf_1/MatMulMatMul&ppo_agent/ppo2_model/flatten_2/Reshapeppo_agent/ppo2_model/vf/w/read*
transpose_a( *
transpose_b( *
T0*
_output_shapes
:	�
�
ppo_agent/ppo2_model/vf_1/addAdd ppo_agent/ppo2_model/vf_1/MatMulppo_agent/ppo2_model/vf/b/read*
T0*
_output_shapes
:	�
{
*ppo_agent/ppo2_model/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
}
,ppo_agent/ppo2_model/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
}
,ppo_agent/ppo2_model/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
$ppo_agent/ppo2_model/strided_slice_1StridedSliceppo_agent/ppo2_model/vf_1/add*ppo_agent/ppo2_model/strided_slice_1/stack,ppo_agent/ppo2_model/strided_slice_1/stack_1,ppo_agent/ppo2_model/strided_slice_1/stack_2*
end_mask*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
_output_shapes	
:�
t
ppo_agent/ppo2_model/value_1Identity$ppo_agent/ppo2_model/strided_slice_1*
T0*
_output_shapes	
:�
f
PlaceholderPlaceholder*
shape:���������*
dtype0*#
_output_shapes
:���������
h
Placeholder_1Placeholder*
dtype0*
shape:���������*#
_output_shapes
:���������
h
Placeholder_2Placeholder*
shape:���������*
dtype0*#
_output_shapes
:���������
h
Placeholder_3Placeholder*
dtype0*
shape:���������*#
_output_shapes
:���������
h
Placeholder_4Placeholder*
dtype0*
shape:���������*#
_output_shapes
:���������
N
Placeholder_5Placeholder*
dtype0*
shape: *
_output_shapes
: 
N
Placeholder_6Placeholder*
dtype0*
shape: *
_output_shapes
: 
U
one_hot/on_valueConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
V
one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
O
one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
�
one_hotOneHotPlaceholderone_hot/depthone_hot/on_valueone_hot/off_value*
T0*
TI0*
axis���������*'
_output_shapes
:���������
h
&softmax_cross_entropy_with_logits/RankConst*
dtype0*
value	B :*
_output_shapes
: 
x
'softmax_cross_entropy_with_logits/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
j
(softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
z
)softmax_cross_entropy_with_logits/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
i
'softmax_cross_entropy_with_logits/Sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
%softmax_cross_entropy_with_logits/SubSub(softmax_cross_entropy_with_logits/Rank_1'softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
�
-softmax_cross_entropy_with_logits/Slice/beginPack%softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N*
_output_shapes
:
v
,softmax_cross_entropy_with_logits/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
�
'softmax_cross_entropy_with_logits/SliceSlice)softmax_cross_entropy_with_logits/Shape_1-softmax_cross_entropy_with_logits/Slice/begin,softmax_cross_entropy_with_logits/Slice/size*
T0*
Index0*
_output_shapes
:
�
1softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
o
-softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
(softmax_cross_entropy_with_logits/concatConcatV21softmax_cross_entropy_with_logits/concat/values_0'softmax_cross_entropy_with_logits/Slice-softmax_cross_entropy_with_logits/concat/axis*
T0*
N*

Tidx0*
_output_shapes
:
�
)softmax_cross_entropy_with_logits/ReshapeReshapeppo_agent/ppo2_model/pi_3/add(softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*
_output_shapes
:	�
j
(softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
p
)softmax_cross_entropy_with_logits/Shape_2Shapeone_hot*
T0*
out_type0*
_output_shapes
:
k
)softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'softmax_cross_entropy_with_logits/Sub_1Sub(softmax_cross_entropy_with_logits/Rank_2)softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
�
/softmax_cross_entropy_with_logits/Slice_1/beginPack'softmax_cross_entropy_with_logits/Sub_1*
T0*

axis *
N*
_output_shapes
:
x
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
)softmax_cross_entropy_with_logits/Slice_1Slice)softmax_cross_entropy_with_logits/Shape_2/softmax_cross_entropy_with_logits/Slice_1/begin.softmax_cross_entropy_with_logits/Slice_1/size*
T0*
Index0*
_output_shapes
:
�
3softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
q
/softmax_cross_entropy_with_logits/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
*softmax_cross_entropy_with_logits/concat_1ConcatV23softmax_cross_entropy_with_logits/concat_1/values_0)softmax_cross_entropy_with_logits/Slice_1/softmax_cross_entropy_with_logits/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
+softmax_cross_entropy_with_logits/Reshape_1Reshapeone_hot*softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits)softmax_cross_entropy_with_logits/Reshape+softmax_cross_entropy_with_logits/Reshape_1*
T0*&
_output_shapes
:�:	�
k
)softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'softmax_cross_entropy_with_logits/Sub_2Sub&softmax_cross_entropy_with_logits/Rank)softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
y
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
.softmax_cross_entropy_with_logits/Slice_2/sizePack'softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
)softmax_cross_entropy_with_logits/Slice_2Slice'softmax_cross_entropy_with_logits/Shape/softmax_cross_entropy_with_logits/Slice_2/begin.softmax_cross_entropy_with_logits/Slice_2/size*
T0*
Index0*
_output_shapes
:
�
+softmax_cross_entropy_with_logits/Reshape_2Reshape!softmax_cross_entropy_with_logits)softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*
_output_shapes	
:�
`
Max/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
MaxMaxppo_agent/ppo2_model/pi_3/addMax/reduction_indices*
T0*

Tidx0*
	keep_dims(*
_output_shapes
:	�
X
subSubppo_agent/ppo2_model/pi_3/addMax*
T0*
_output_shapes
:	�
9
ExpExpsub*
T0*
_output_shapes
:	�
`
Sum/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
m
SumSumExpSum/reduction_indices*

Tidx0*
	keep_dims(*
T0*
_output_shapes
:	�
F
truedivRealDivExpSum*
T0*
_output_shapes
:	�
9
LogLogSum*
T0*
_output_shapes
:	�
@
sub_1SubLogsub*
T0*
_output_shapes
:	�
D
mulMultruedivsub_1*
T0*
_output_shapes
:	�
b
Sum_1/reduction_indicesConst*
dtype0*
valueB :
���������*
_output_shapes
: 
m
Sum_1SummulSum_1/reduction_indices*
T0*

Tidx0*
	keep_dims( *
_output_shapes	
:�
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
X
MeanMeanSum_1Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
_
sub_2Subppo_agent/ppo2_model/value_1Placeholder_4*
T0*
_output_shapes	
:�
:
NegNegPlaceholder_6*
T0*
_output_shapes
: 
\
clip_by_value/MinimumMinimumsub_2Placeholder_6*
T0*
_output_shapes	
:�
Z
clip_by_valueMaximumclip_by_value/MinimumNeg*
T0*
_output_shapes	
:�
N
addAddPlaceholder_4clip_by_value*
T0*
_output_shapes	
:�
_
sub_3Subppo_agent/ppo2_model/value_1Placeholder_2*
T0*
_output_shapes	
:�
=
SquareSquaresub_3*
T0*
_output_shapes	
:�
F
sub_4SubaddPlaceholder_2*
T0*
_output_shapes	
:�
?
Square_1Squaresub_4*
T0*
_output_shapes	
:�
J
MaximumMaximumSquareSquare_1*
T0*
_output_shapes	
:�
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
^
Mean_1MeanMaximumConst_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
L
mul_1/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
>
mul_1Mulmul_1/xMean_1*
T0*
_output_shapes
: 
n
sub_5SubPlaceholder_3+softmax_cross_entropy_with_logits/Reshape_2*
T0*
_output_shapes	
:�
9
Exp_1Expsub_5*
T0*
_output_shapes	
:�
I
Neg_1NegPlaceholder_1*
T0*#
_output_shapes
:���������
@
mul_2MulNeg_1Exp_1*
T0*
_output_shapes	
:�
I
Neg_2NegPlaceholder_1*
T0*#
_output_shapes
:���������
L
sub_6/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
E
sub_6Subsub_6/xPlaceholder_6*
T0*
_output_shapes
: 
L
add_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
E
add_1Addadd_1/xPlaceholder_6*
T0*
_output_shapes
: 
V
clip_by_value_1/MinimumMinimumExp_1add_1*
T0*
_output_shapes	
:�
`
clip_by_value_1Maximumclip_by_value_1/Minimumsub_6*
T0*
_output_shapes	
:�
J
mul_3MulNeg_2clip_by_value_1*
T0*
_output_shapes	
:�
H
	Maximum_1Maximummul_2mul_3*
T0*
_output_shapes	
:�
Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:
`
Mean_2Mean	Maximum_1Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
n
sub_7Sub+softmax_cross_entropy_with_logits/Reshape_2Placeholder_3*
T0*
_output_shapes	
:�
?
Square_2Squaresub_7*
T0*
_output_shapes	
:�
Q
Const_3Const*
valueB: *
dtype0*
_output_shapes
:
_
Mean_3MeanSquare_2Const_3*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
L
mul_4/xConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
>
mul_4Mulmul_4/xMean_3*
T0*
_output_shapes
: 
L
sub_8/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
B
sub_8SubExp_1sub_8/y*
T0*
_output_shapes	
:�
7
AbsAbssub_8*
T0*
_output_shapes	
:�
L
GreaterGreaterAbsPlaceholder_6*
T0*
_output_shapes	
:�
]
ToFloatCastGreater*

SrcT0
*
Truncate( *

DstT0*
_output_shapes	
:�
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
^
Mean_4MeanToFloatConst_4*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
L
mul_5/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
<
mul_5MulMeanmul_5/y*
T0*
_output_shapes
: 
<
sub_9SubMean_2mul_5*
T0*
_output_shapes
: 
L
mul_6/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
=
mul_6Mulmul_1mul_6/y*
T0*
_output_shapes
: 
;
add_2Addsub_9mul_6*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
>
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/Fill
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/Fill&^gradients/add_2_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/Fill&^gradients/add_2_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
o
gradients/sub_9_grad/NegNeg-gradients/add_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
x
%gradients/sub_9_grad/tuple/group_depsNoOp.^gradients/add_2_grad/tuple/control_dependency^gradients/sub_9_grad/Neg
�
-gradients/sub_9_grad/tuple/control_dependencyIdentity-gradients/add_2_grad/tuple/control_dependency&^gradients/sub_9_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
/gradients/sub_9_grad/tuple/control_dependency_1Identitygradients/sub_9_grad/Neg&^gradients/sub_9_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/sub_9_grad/Neg*
_output_shapes
: 
z
gradients/mul_6_grad/MulMul/gradients/add_2_grad/tuple/control_dependency_1mul_6/y*
T0*
_output_shapes
: 
z
gradients/mul_6_grad/Mul_1Mul/gradients/add_2_grad/tuple/control_dependency_1mul_1*
T0*
_output_shapes
: 
e
%gradients/mul_6_grad/tuple/group_depsNoOp^gradients/mul_6_grad/Mul^gradients/mul_6_grad/Mul_1
�
-gradients/mul_6_grad/tuple/control_dependencyIdentitygradients/mul_6_grad/Mul&^gradients/mul_6_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_6_grad/Mul*
_output_shapes
: 
�
/gradients/mul_6_grad/tuple/control_dependency_1Identitygradients/mul_6_grad/Mul_1&^gradients/mul_6_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_6_grad/Mul_1*
_output_shapes
: 
m
#gradients/Mean_2_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_2_grad/ReshapeReshape-gradients/sub_9_grad/tuple/control_dependency#gradients/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
f
gradients/Mean_2_grad/ConstConst*
dtype0*
valueB:�*
_output_shapes
:
�
gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Const*
T0*

Tmultiples0*
_output_shapes	
:�
b
gradients/Mean_2_grad/Const_1Const*
valueB
 *  HD*
dtype0*
_output_shapes
: 
�
gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Const_1*
T0*
_output_shapes	
:�
z
gradients/mul_5_grad/MulMul/gradients/sub_9_grad/tuple/control_dependency_1mul_5/y*
T0*
_output_shapes
: 
y
gradients/mul_5_grad/Mul_1Mul/gradients/sub_9_grad/tuple/control_dependency_1Mean*
T0*
_output_shapes
: 
e
%gradients/mul_5_grad/tuple/group_depsNoOp^gradients/mul_5_grad/Mul^gradients/mul_5_grad/Mul_1
�
-gradients/mul_5_grad/tuple/control_dependencyIdentitygradients/mul_5_grad/Mul&^gradients/mul_5_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_5_grad/Mul*
_output_shapes
: 
�
/gradients/mul_5_grad/tuple/control_dependency_1Identitygradients/mul_5_grad/Mul_1&^gradients/mul_5_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_5_grad/Mul_1*
_output_shapes
: 
w
gradients/mul_1_grad/MulMul-gradients/mul_6_grad/tuple/control_dependencyMean_1*
T0*
_output_shapes
: 
z
gradients/mul_1_grad/Mul_1Mul-gradients/mul_6_grad/tuple/control_dependencymul_1/x*
T0*
_output_shapes
: 
e
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Mul^gradients/mul_1_grad/Mul_1
�
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Mul&^gradients/mul_1_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_1_grad/Mul*
_output_shapes
: 
�
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Mul_1&^gradients/mul_1_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_1_grad/Mul_1*
_output_shapes
: 
i
gradients/Maximum_1_grad/ShapeConst*
dtype0*
valueB:�*
_output_shapes
:
k
 gradients/Maximum_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
k
 gradients/Maximum_1_grad/Shape_2Const*
valueB:�*
dtype0*
_output_shapes
:
i
$gradients/Maximum_1_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients/Maximum_1_grad/zerosFill gradients/Maximum_1_grad/Shape_2$gradients/Maximum_1_grad/zeros/Const*
T0*

index_type0*
_output_shapes	
:�
i
%gradients/Maximum_1_grad/GreaterEqualGreaterEqualmul_2mul_3*
T0*
_output_shapes	
:�
�
.gradients/Maximum_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_1_grad/Shape gradients/Maximum_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Maximum_1_grad/SelectSelect%gradients/Maximum_1_grad/GreaterEqualgradients/Mean_2_grad/truedivgradients/Maximum_1_grad/zeros*
T0*
_output_shapes	
:�
�
!gradients/Maximum_1_grad/Select_1Select%gradients/Maximum_1_grad/GreaterEqualgradients/Maximum_1_grad/zerosgradients/Mean_2_grad/truediv*
T0*
_output_shapes	
:�
�
gradients/Maximum_1_grad/SumSumgradients/Maximum_1_grad/Select.gradients/Maximum_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes	
:�
�
 gradients/Maximum_1_grad/ReshapeReshapegradients/Maximum_1_grad/Sumgradients/Maximum_1_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
gradients/Maximum_1_grad/Sum_1Sum!gradients/Maximum_1_grad/Select_10gradients/Maximum_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:�
�
"gradients/Maximum_1_grad/Reshape_1Reshapegradients/Maximum_1_grad/Sum_1 gradients/Maximum_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
y
)gradients/Maximum_1_grad/tuple/group_depsNoOp!^gradients/Maximum_1_grad/Reshape#^gradients/Maximum_1_grad/Reshape_1
�
1gradients/Maximum_1_grad/tuple/control_dependencyIdentity gradients/Maximum_1_grad/Reshape*^gradients/Maximum_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Maximum_1_grad/Reshape*
_output_shapes	
:�
�
3gradients/Maximum_1_grad/tuple/control_dependency_1Identity"gradients/Maximum_1_grad/Reshape_1*^gradients/Maximum_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/Maximum_1_grad/Reshape_1*
_output_shapes	
:�
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshape-gradients/mul_5_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
d
gradients/Mean_grad/ConstConst*
dtype0*
valueB:�*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Const*

Tmultiples0*
T0*
_output_shapes	
:�
`
gradients/Mean_grad/Const_1Const*
valueB
 *  HD*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Const_1*
T0*
_output_shapes	
:�
m
#gradients/Mean_1_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Mean_1_grad/ReshapeReshape/gradients/mul_1_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
f
gradients/Mean_1_grad/ConstConst*
valueB:�*
dtype0*
_output_shapes
:
�
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Const*

Tmultiples0*
T0*
_output_shapes	
:�
b
gradients/Mean_1_grad/Const_1Const*
valueB
 *  HD*
dtype0*
_output_shapes
: 
�
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Const_1*
T0*
_output_shapes	
:�
_
gradients/mul_2_grad/ShapeShapeNeg_1*
T0*
out_type0*
_output_shapes
:
g
gradients/mul_2_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������

gradients/mul_2_grad/MulMul1gradients/Maximum_1_grad/tuple/control_dependencyExp_1*
T0*
_output_shapes	
:�
�
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/mul_2_grad/Mul_1MulNeg_11gradients/Maximum_1_grad/tuple/control_dependency*
T0*
_output_shapes	
:�
�
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
�
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*#
_output_shapes
:���������
�
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
_output_shapes	
:�
_
gradients/mul_3_grad/ShapeShapeNeg_2*
T0*
out_type0*
_output_shapes
:
g
gradients/mul_3_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
*gradients/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_3_grad/Shapegradients/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/mul_3_grad/MulMul3gradients/Maximum_1_grad/tuple/control_dependency_1clip_by_value_1*
T0*
_output_shapes	
:�
�
gradients/mul_3_grad/SumSumgradients/mul_3_grad/Mul*gradients/mul_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/mul_3_grad/Mul_1MulNeg_23gradients/Maximum_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/Mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
m
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Reshape^gradients/mul_3_grad/Reshape_1
�
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*#
_output_shapes
:���������
�
/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Reshape_1&^gradients/mul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1*
_output_shapes	
:�
k
gradients/Sum_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients/Sum_1_grad/SizeConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
�
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
�
gradients/Sum_1_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
 gradients/Sum_1_grad/range/startConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
 gradients/Sum_1_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*-
_class#
!loc:@gradients/Sum_1_grad/Shape*

Tidx0*
_output_shapes
:
�
gradients/Sum_1_grad/Fill/valueConst*
dtype0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
_output_shapes
: 
�
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*

index_type0*
_output_shapes
: 
�
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_1_grad/Maximum/yConst*
dtype0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
_output_shapes
: 
�
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
�
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
�
gradients/Sum_1_grad/ReshapeReshapegradients/Mean_grad/truediv"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:	�
g
gradients/Maximum_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
i
gradients/Maximum_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
i
gradients/Maximum_grad/Shape_2Const*
dtype0*
valueB:�*
_output_shapes
:
g
"gradients/Maximum_grad/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
gradients/Maximum_grad/zerosFillgradients/Maximum_grad/Shape_2"gradients/Maximum_grad/zeros/Const*
T0*

index_type0*
_output_shapes	
:�
k
#gradients/Maximum_grad/GreaterEqualGreaterEqualSquareSquare_1*
T0*
_output_shapes	
:�
�
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Maximum_grad/SelectSelect#gradients/Maximum_grad/GreaterEqualgradients/Mean_1_grad/truedivgradients/Maximum_grad/zeros*
T0*
_output_shapes	
:�
�
gradients/Maximum_grad/Select_1Select#gradients/Maximum_grad/GreaterEqualgradients/Maximum_grad/zerosgradients/Mean_1_grad/truediv*
T0*
_output_shapes	
:�
�
gradients/Maximum_grad/SumSumgradients/Maximum_grad/Select,gradients/Maximum_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes	
:�
�
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes	
:�
�
 gradients/Maximum_grad/Reshape_1Reshapegradients/Maximum_grad/Sum_1gradients/Maximum_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
s
'gradients/Maximum_grad/tuple/group_depsNoOp^gradients/Maximum_grad/Reshape!^gradients/Maximum_grad/Reshape_1
�
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Maximum_grad/Reshape*
_output_shapes	
:�
�
1gradients/Maximum_grad/tuple/control_dependency_1Identity gradients/Maximum_grad/Reshape_1(^gradients/Maximum_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Maximum_grad/Reshape_1*
_output_shapes	
:�
o
$gradients/clip_by_value_1_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
i
&gradients/clip_by_value_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
q
&gradients/clip_by_value_1_grad/Shape_2Const*
valueB:�*
dtype0*
_output_shapes
:
o
*gradients/clip_by_value_1_grad/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
$gradients/clip_by_value_1_grad/zerosFill&gradients/clip_by_value_1_grad/Shape_2*gradients/clip_by_value_1_grad/zeros/Const*
T0*

index_type0*
_output_shapes	
:�
�
+gradients/clip_by_value_1_grad/GreaterEqualGreaterEqualclip_by_value_1/Minimumsub_6*
T0*
_output_shapes	
:�
�
4gradients/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/clip_by_value_1_grad/Shape&gradients/clip_by_value_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
%gradients/clip_by_value_1_grad/SelectSelect+gradients/clip_by_value_1_grad/GreaterEqual/gradients/mul_3_grad/tuple/control_dependency_1$gradients/clip_by_value_1_grad/zeros*
T0*
_output_shapes	
:�
�
'gradients/clip_by_value_1_grad/Select_1Select+gradients/clip_by_value_1_grad/GreaterEqual$gradients/clip_by_value_1_grad/zeros/gradients/mul_3_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
"gradients/clip_by_value_1_grad/SumSum%gradients/clip_by_value_1_grad/Select4gradients/clip_by_value_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes	
:�
�
&gradients/clip_by_value_1_grad/ReshapeReshape"gradients/clip_by_value_1_grad/Sum$gradients/clip_by_value_1_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
$gradients/clip_by_value_1_grad/Sum_1Sum'gradients/clip_by_value_1_grad/Select_16gradients/clip_by_value_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
(gradients/clip_by_value_1_grad/Reshape_1Reshape$gradients/clip_by_value_1_grad/Sum_1&gradients/clip_by_value_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
/gradients/clip_by_value_1_grad/tuple/group_depsNoOp'^gradients/clip_by_value_1_grad/Reshape)^gradients/clip_by_value_1_grad/Reshape_1
�
7gradients/clip_by_value_1_grad/tuple/control_dependencyIdentity&gradients/clip_by_value_1_grad/Reshape0^gradients/clip_by_value_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/clip_by_value_1_grad/Reshape*
_output_shapes	
:�
�
9gradients/clip_by_value_1_grad/tuple/control_dependency_1Identity(gradients/clip_by_value_1_grad/Reshape_10^gradients/clip_by_value_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/clip_by_value_1_grad/Reshape_1*
_output_shapes
: 
i
gradients/mul_grad/MulMulgradients/Sum_1_grad/Tilesub_1*
T0*
_output_shapes
:	�
m
gradients/mul_grad/Mul_1Mulgradients/Sum_1_grad/Tiletruediv*
T0*
_output_shapes
:	�
_
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Mul^gradients/mul_grad/Mul_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Mul$^gradients/mul_grad/tuple/group_deps*
T0*)
_class
loc:@gradients/mul_grad/Mul*
_output_shapes
:	�
�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Mul_1$^gradients/mul_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_grad/Mul_1*
_output_shapes
:	�
�
gradients/Square_grad/ConstConst0^gradients/Maximum_grad/tuple/control_dependency*
valueB
 *   @*
dtype0*
_output_shapes
: 
j
gradients/Square_grad/MulMulsub_3gradients/Square_grad/Const*
T0*
_output_shapes	
:�
�
gradients/Square_grad/Mul_1Mul/gradients/Maximum_grad/tuple/control_dependencygradients/Square_grad/Mul*
T0*
_output_shapes	
:�
�
gradients/Square_1_grad/ConstConst2^gradients/Maximum_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
n
gradients/Square_1_grad/MulMulsub_4gradients/Square_1_grad/Const*
T0*
_output_shapes	
:�
�
gradients/Square_1_grad/Mul_1Mul1gradients/Maximum_grad/tuple/control_dependency_1gradients/Square_1_grad/Mul*
T0*
_output_shapes	
:�
w
,gradients/clip_by_value_1/Minimum_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
q
.gradients/clip_by_value_1/Minimum_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
y
.gradients/clip_by_value_1/Minimum_grad/Shape_2Const*
dtype0*
valueB:�*
_output_shapes
:
w
2gradients/clip_by_value_1/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
,gradients/clip_by_value_1/Minimum_grad/zerosFill.gradients/clip_by_value_1/Minimum_grad/Shape_22gradients/clip_by_value_1/Minimum_grad/zeros/Const*
T0*

index_type0*
_output_shapes	
:�
q
0gradients/clip_by_value_1/Minimum_grad/LessEqual	LessEqualExp_1add_1*
T0*
_output_shapes	
:�
�
<gradients/clip_by_value_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/clip_by_value_1/Minimum_grad/Shape.gradients/clip_by_value_1/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
-gradients/clip_by_value_1/Minimum_grad/SelectSelect0gradients/clip_by_value_1/Minimum_grad/LessEqual7gradients/clip_by_value_1_grad/tuple/control_dependency,gradients/clip_by_value_1/Minimum_grad/zeros*
T0*
_output_shapes	
:�
�
/gradients/clip_by_value_1/Minimum_grad/Select_1Select0gradients/clip_by_value_1/Minimum_grad/LessEqual,gradients/clip_by_value_1/Minimum_grad/zeros7gradients/clip_by_value_1_grad/tuple/control_dependency*
T0*
_output_shapes	
:�
�
*gradients/clip_by_value_1/Minimum_grad/SumSum-gradients/clip_by_value_1/Minimum_grad/Select<gradients/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:�
�
.gradients/clip_by_value_1/Minimum_grad/ReshapeReshape*gradients/clip_by_value_1/Minimum_grad/Sum,gradients/clip_by_value_1/Minimum_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
,gradients/clip_by_value_1/Minimum_grad/Sum_1Sum/gradients/clip_by_value_1/Minimum_grad/Select_1>gradients/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
�
0gradients/clip_by_value_1/Minimum_grad/Reshape_1Reshape,gradients/clip_by_value_1/Minimum_grad/Sum_1.gradients/clip_by_value_1/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
7gradients/clip_by_value_1/Minimum_grad/tuple/group_depsNoOp/^gradients/clip_by_value_1/Minimum_grad/Reshape1^gradients/clip_by_value_1/Minimum_grad/Reshape_1
�
?gradients/clip_by_value_1/Minimum_grad/tuple/control_dependencyIdentity.gradients/clip_by_value_1/Minimum_grad/Reshape8^gradients/clip_by_value_1/Minimum_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/clip_by_value_1/Minimum_grad/Reshape*
_output_shapes	
:�
�
Agradients/clip_by_value_1/Minimum_grad/tuple/control_dependency_1Identity0gradients/clip_by_value_1/Minimum_grad/Reshape_18^gradients/clip_by_value_1/Minimum_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/clip_by_value_1/Minimum_grad/Reshape_1*
_output_shapes
: 
m
gradients/truediv_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
o
gradients/truediv_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/truediv_grad/RealDivRealDiv+gradients/mul_grad/tuple/control_dependencySum*
T0*
_output_shapes
:	�
�
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:	�
�
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
P
gradients/truediv_grad/NegNegExp*
T0*
_output_shapes
:	�
v
 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegSum*
T0*
_output_shapes
:	�
|
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Sum*
T0*
_output_shapes
:	�
�
gradients/truediv_grad/mulMul+gradients/mul_grad/tuple/control_dependency gradients/truediv_grad/RealDiv_2*
T0*
_output_shapes
:	�
�
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes	
:�
�
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:	�
s
'gradients/truediv_grad/tuple/group_depsNoOp^gradients/truediv_grad/Reshape!^gradients/truediv_grad/Reshape_1
�
/gradients/truediv_grad/tuple/control_dependencyIdentitygradients/truediv_grad/Reshape(^gradients/truediv_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/truediv_grad/Reshape*
_output_shapes
:	�
�
1gradients/truediv_grad/tuple/control_dependency_1Identity gradients/truediv_grad/Reshape_1(^gradients/truediv_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_grad/Reshape_1*
_output_shapes
:	�
k
gradients/sub_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
gradients/sub_1_grad/Shape_1Const*
dtype0*
valueB"      *
_output_shapes
:
�
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_1_grad/SumSum-gradients/mul_grad/tuple/control_dependency_1*gradients/sub_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes	
:�
�
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/sub_1_grad/Sum_1Sum-gradients/mul_grad/tuple/control_dependency_1,gradients/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:	�
e
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
T0*
_output_shapes
:	�
�
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:	�
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
�
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape*
_output_shapes
:	�
�
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1*
_output_shapes
:	�
e
gradients/sub_3_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
i
gradients/sub_3_grad/Shape_1ShapePlaceholder_2*
T0*
out_type0*
_output_shapes
:
�
*gradients/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_3_grad/Shapegradients/sub_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_3_grad/SumSumgradients/Square_grad/Mul_1*gradients/sub_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/sub_3_grad/ReshapeReshapegradients/sub_3_grad/Sumgradients/sub_3_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
gradients/sub_3_grad/Sum_1Sumgradients/Square_grad/Mul_1,gradients/sub_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
^
gradients/sub_3_grad/NegNeggradients/sub_3_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_3_grad/Reshape_1Reshapegradients/sub_3_grad/Neggradients/sub_3_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
m
%gradients/sub_3_grad/tuple/group_depsNoOp^gradients/sub_3_grad/Reshape^gradients/sub_3_grad/Reshape_1
�
-gradients/sub_3_grad/tuple/control_dependencyIdentitygradients/sub_3_grad/Reshape&^gradients/sub_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_3_grad/Reshape*
_output_shapes	
:�
�
/gradients/sub_3_grad/tuple/control_dependency_1Identitygradients/sub_3_grad/Reshape_1&^gradients/sub_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_3_grad/Reshape_1*#
_output_shapes
:���������
e
gradients/sub_4_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
i
gradients/sub_4_grad/Shape_1ShapePlaceholder_2*
T0*
out_type0*
_output_shapes
:
�
*gradients/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_4_grad/Shapegradients/sub_4_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_4_grad/SumSumgradients/Square_1_grad/Mul_1*gradients/sub_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/sub_4_grad/ReshapeReshapegradients/sub_4_grad/Sumgradients/sub_4_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
gradients/sub_4_grad/Sum_1Sumgradients/Square_1_grad/Mul_1,gradients/sub_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
^
gradients/sub_4_grad/NegNeggradients/sub_4_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_4_grad/Reshape_1Reshapegradients/sub_4_grad/Neggradients/sub_4_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
m
%gradients/sub_4_grad/tuple/group_depsNoOp^gradients/sub_4_grad/Reshape^gradients/sub_4_grad/Reshape_1
�
-gradients/sub_4_grad/tuple/control_dependencyIdentitygradients/sub_4_grad/Reshape&^gradients/sub_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_4_grad/Reshape*
_output_shapes	
:�
�
/gradients/sub_4_grad/tuple/control_dependency_1Identitygradients/sub_4_grad/Reshape_1&^gradients/sub_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_4_grad/Reshape_1*#
_output_shapes
:���������
�
gradients/AddNAddN/gradients/mul_2_grad/tuple/control_dependency_1?gradients/clip_by_value_1/Minimum_grad/tuple/control_dependency*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
N*
_output_shapes	
:�
\
gradients/Exp_1_grad/mulMulgradients/AddNExp_1*
T0*
_output_shapes	
:�
�
gradients/Log_grad/Reciprocal
ReciprocalSum.^gradients/sub_1_grad/tuple/control_dependency*
T0*
_output_shapes
:	�
�
gradients/Log_grad/mulMul-gradients/sub_1_grad/tuple/control_dependencygradients/Log_grad/Reciprocal*
T0*
_output_shapes
:	�
e
gradients/add_grad/ShapeShapePlaceholder_4*
T0*
out_type0*
_output_shapes
:
e
gradients/add_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSum-gradients/sub_4_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sum-gradients/sub_4_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*#
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes	
:�
g
gradients/sub_5_grad/ShapeShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
g
gradients/sub_5_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/sub_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_5_grad/Shapegradients/sub_5_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_5_grad/SumSumgradients/Exp_1_grad/mul*gradients/sub_5_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
gradients/sub_5_grad/ReshapeReshapegradients/sub_5_grad/Sumgradients/sub_5_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/sub_5_grad/Sum_1Sumgradients/Exp_1_grad/mul,gradients/sub_5_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
^
gradients/sub_5_grad/NegNeggradients/sub_5_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_5_grad/Reshape_1Reshapegradients/sub_5_grad/Neggradients/sub_5_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
m
%gradients/sub_5_grad/tuple/group_depsNoOp^gradients/sub_5_grad/Reshape^gradients/sub_5_grad/Reshape_1
�
-gradients/sub_5_grad/tuple/control_dependencyIdentitygradients/sub_5_grad/Reshape&^gradients/sub_5_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_5_grad/Reshape*#
_output_shapes
:���������
�
/gradients/sub_5_grad/tuple/control_dependency_1Identitygradients/sub_5_grad/Reshape_1&^gradients/sub_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_5_grad/Reshape_1*
_output_shapes	
:�
�
gradients/AddN_1AddN1gradients/truediv_grad/tuple/control_dependency_1gradients/Log_grad/mul*
T0*3
_class)
'%loc:@gradients/truediv_grad/Reshape_1*
N*
_output_shapes
:	�
i
gradients/Sum_grad/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
�
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
�
gradients/Sum_grad/Shape_1Const*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
valueB *
_output_shapes
: 
�
gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*+
_class!
loc:@gradients/Sum_grad/Shape*

Tidx0*
_output_shapes
:
�
gradients/Sum_grad/Fill/valueConst*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*

index_type0*
_output_shapes
: 
�
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N*
_output_shapes
:
�
gradients/Sum_grad/Maximum/yConst*
dtype0*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
�
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
�
gradients/Sum_grad/ReshapeReshapegradients/AddN_1 gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:	�
m
"gradients/clip_by_value_grad/ShapeConst*
dtype0*
valueB:�*
_output_shapes
:
g
$gradients/clip_by_value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
o
$gradients/clip_by_value_grad/Shape_2Const*
dtype0*
valueB:�*
_output_shapes
:
m
(gradients/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"gradients/clip_by_value_grad/zerosFill$gradients/clip_by_value_grad/Shape_2(gradients/clip_by_value_grad/zeros/Const*
T0*

index_type0*
_output_shapes	
:�
{
)gradients/clip_by_value_grad/GreaterEqualGreaterEqualclip_by_value/MinimumNeg*
T0*
_output_shapes	
:�
�
2gradients/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/clip_by_value_grad/Shape$gradients/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#gradients/clip_by_value_grad/SelectSelect)gradients/clip_by_value_grad/GreaterEqual-gradients/add_grad/tuple/control_dependency_1"gradients/clip_by_value_grad/zeros*
T0*
_output_shapes	
:�
�
%gradients/clip_by_value_grad/Select_1Select)gradients/clip_by_value_grad/GreaterEqual"gradients/clip_by_value_grad/zeros-gradients/add_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
 gradients/clip_by_value_grad/SumSum#gradients/clip_by_value_grad/Select2gradients/clip_by_value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:�
�
$gradients/clip_by_value_grad/ReshapeReshape gradients/clip_by_value_grad/Sum"gradients/clip_by_value_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
"gradients/clip_by_value_grad/Sum_1Sum%gradients/clip_by_value_grad/Select_14gradients/clip_by_value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
&gradients/clip_by_value_grad/Reshape_1Reshape"gradients/clip_by_value_grad/Sum_1$gradients/clip_by_value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
-gradients/clip_by_value_grad/tuple/group_depsNoOp%^gradients/clip_by_value_grad/Reshape'^gradients/clip_by_value_grad/Reshape_1
�
5gradients/clip_by_value_grad/tuple/control_dependencyIdentity$gradients/clip_by_value_grad/Reshape.^gradients/clip_by_value_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/clip_by_value_grad/Reshape*
_output_shapes	
:�
�
7gradients/clip_by_value_grad/tuple/control_dependency_1Identity&gradients/clip_by_value_grad/Reshape_1.^gradients/clip_by_value_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/clip_by_value_grad/Reshape_1*
_output_shapes
: 
�
@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
Bgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshape/gradients/sub_5_grad/tuple/control_dependency_1@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
gradients/AddN_2AddN/gradients/truediv_grad/tuple/control_dependencygradients/Sum_grad/Tile*
T0*1
_class'
%#loc:@gradients/truediv_grad/Reshape*
N*
_output_shapes
:	�
^
gradients/Exp_grad/mulMulgradients/AddN_2Exp*
T0*
_output_shapes
:	�
u
*gradients/clip_by_value/Minimum_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
o
,gradients/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
w
,gradients/clip_by_value/Minimum_grad/Shape_2Const*
valueB:�*
dtype0*
_output_shapes
:
u
0gradients/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
*gradients/clip_by_value/Minimum_grad/zerosFill,gradients/clip_by_value/Minimum_grad/Shape_20gradients/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0*
_output_shapes	
:�
w
.gradients/clip_by_value/Minimum_grad/LessEqual	LessEqualsub_2Placeholder_6*
T0*
_output_shapes	
:�
�
:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/clip_by_value/Minimum_grad/Shape,gradients/clip_by_value/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
+gradients/clip_by_value/Minimum_grad/SelectSelect.gradients/clip_by_value/Minimum_grad/LessEqual5gradients/clip_by_value_grad/tuple/control_dependency*gradients/clip_by_value/Minimum_grad/zeros*
T0*
_output_shapes	
:�
�
-gradients/clip_by_value/Minimum_grad/Select_1Select.gradients/clip_by_value/Minimum_grad/LessEqual*gradients/clip_by_value/Minimum_grad/zeros5gradients/clip_by_value_grad/tuple/control_dependency*
T0*
_output_shapes	
:�
�
(gradients/clip_by_value/Minimum_grad/SumSum+gradients/clip_by_value/Minimum_grad/Select:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes	
:�
�
,gradients/clip_by_value/Minimum_grad/ReshapeReshape(gradients/clip_by_value/Minimum_grad/Sum*gradients/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
*gradients/clip_by_value/Minimum_grad/Sum_1Sum-gradients/clip_by_value/Minimum_grad/Select_1<gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
�
.gradients/clip_by_value/Minimum_grad/Reshape_1Reshape*gradients/clip_by_value/Minimum_grad/Sum_1,gradients/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
5gradients/clip_by_value/Minimum_grad/tuple/group_depsNoOp-^gradients/clip_by_value/Minimum_grad/Reshape/^gradients/clip_by_value/Minimum_grad/Reshape_1
�
=gradients/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity,gradients/clip_by_value/Minimum_grad/Reshape6^gradients/clip_by_value/Minimum_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/clip_by_value/Minimum_grad/Reshape*
_output_shapes	
:�
�
?gradients/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity.gradients/clip_by_value/Minimum_grad/Reshape_16^gradients/clip_by_value/Minimum_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/clip_by_value/Minimum_grad/Reshape_1*
_output_shapes
: 
p
gradients/zeros_like	ZerosLike#softmax_cross_entropy_with_logits:1*
T0*
_output_shapes
:	�
�
?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:	�
�
4gradients/softmax_cross_entropy_with_logits_grad/mulMul;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims#softmax_cross_entropy_with_logits:1*
T0*
_output_shapes
:	�
�
;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax)softmax_cross_entropy_with_logits/Reshape*
T0*
_output_shapes
:	�
�
4gradients/softmax_cross_entropy_with_logits_grad/NegNeg;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0*
_output_shapes
:	�
�
Agradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:	�
�
6gradients/softmax_cross_entropy_with_logits_grad/mul_1Mul=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_14gradients/softmax_cross_entropy_with_logits_grad/Neg*
T0*
_output_shapes
:	�
�
Agradients/softmax_cross_entropy_with_logits_grad/tuple/group_depsNoOp5^gradients/softmax_cross_entropy_with_logits_grad/mul7^gradients/softmax_cross_entropy_with_logits_grad/mul_1
�
Igradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependencyIdentity4gradients/softmax_cross_entropy_with_logits_grad/mulB^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/softmax_cross_entropy_with_logits_grad/mul*
_output_shapes
:	�
�
Kgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1Identity6gradients/softmax_cross_entropy_with_logits_grad/mul_1B^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_with_logits_grad/mul_1*
_output_shapes
:	�
�
gradients/AddN_3AddN/gradients/sub_1_grad/tuple/control_dependency_1gradients/Exp_grad/mul*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1*
N*
_output_shapes
:	�
i
gradients/sub_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
k
gradients/sub_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSumgradients/AddN_3(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:	�
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/sub_grad/Sum_1Sumgradients/AddN_3*gradients/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:�
]
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes	
:�
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:	�
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
:	�
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
_output_shapes
:	�
e
gradients/sub_2_grad/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
i
gradients/sub_2_grad/Shape_1ShapePlaceholder_4*
T0*
out_type0*
_output_shapes
:
�
*gradients/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_2_grad/Shapegradients/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_2_grad/SumSum=gradients/clip_by_value/Minimum_grad/tuple/control_dependency*gradients/sub_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients/sub_2_grad/ReshapeReshapegradients/sub_2_grad/Sumgradients/sub_2_grad/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
gradients/sub_2_grad/Sum_1Sum=gradients/clip_by_value/Minimum_grad/tuple/control_dependency,gradients/sub_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
^
gradients/sub_2_grad/NegNeggradients/sub_2_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_2_grad/Reshape_1Reshapegradients/sub_2_grad/Neggradients/sub_2_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
m
%gradients/sub_2_grad/tuple/group_depsNoOp^gradients/sub_2_grad/Reshape^gradients/sub_2_grad/Reshape_1
�
-gradients/sub_2_grad/tuple/control_dependencyIdentitygradients/sub_2_grad/Reshape&^gradients/sub_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_2_grad/Reshape*
_output_shapes	
:�
�
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Reshape_1&^gradients/sub_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_2_grad/Reshape_1*#
_output_shapes
:���������
�
>gradients/softmax_cross_entropy_with_logits/Reshape_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
@gradients/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapeIgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency>gradients/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
i
gradients/Max_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
Y
gradients/Max_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
n
gradients/Max_grad/addAddMax/reduction_indicesgradients/Max_grad/Size*
T0*
_output_shapes
: 
t
gradients/Max_grad/modFloorModgradients/Max_grad/addgradients/Max_grad/Size*
T0*
_output_shapes
: 
]
gradients/Max_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
`
gradients/Max_grad/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
`
gradients/Max_grad/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/Max_grad/rangeRangegradients/Max_grad/range/startgradients/Max_grad/Sizegradients/Max_grad/range/delta*

Tidx0*
_output_shapes
:
_
gradients/Max_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Max_grad/FillFillgradients/Max_grad/Shape_1gradients/Max_grad/Fill/value*
T0*

index_type0*
_output_shapes
: 
�
 gradients/Max_grad/DynamicStitchDynamicStitchgradients/Max_grad/rangegradients/Max_grad/modgradients/Max_grad/Shapegradients/Max_grad/Fill*
N*
T0*
_output_shapes
:
�
gradients/Max_grad/ReshapeReshapeMax gradients/Max_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/Max_grad/Reshape_1Reshape-gradients/sub_grad/tuple/control_dependency_1 gradients/Max_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/Max_grad/EqualEqualgradients/Max_grad/Reshapeppo_agent/ppo2_model/pi_3/add*
T0*
_output_shapes
:	�
�
gradients/Max_grad/CastCastgradients/Max_grad/Equal*

SrcT0
*
Truncate( *

DstT0*
_output_shapes
:	�
�
gradients/Max_grad/SumSumgradients/Max_grad/CastMax/reduction_indices*
T0*

Tidx0*
	keep_dims( *
_output_shapes	
:�
�
gradients/Max_grad/Reshape_2Reshapegradients/Max_grad/Sum gradients/Max_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:	�
�
gradients/Max_grad/divRealDivgradients/Max_grad/Castgradients/Max_grad/Reshape_2*
T0*
_output_shapes
:	�
}
gradients/Max_grad/mulMulgradients/Max_grad/divgradients/Max_grad/Reshape_1*
T0*
_output_shapes
:	�
�
gradients/AddN_4AddN-gradients/sub_3_grad/tuple/control_dependency-gradients/sub_2_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/sub_3_grad/Reshape*
_output_shapes	
:�
�
gradients/AddN_5AddN+gradients/sub_grad/tuple/control_dependency@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshapegradients/Max_grad/mul*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
N*
_output_shapes
:	�
�
2gradients/ppo_agent/ppo2_model/pi_3/add_grad/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
~
4gradients/ppo_agent/ppo2_model/pi_3/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Bgradients/ppo_agent/ppo2_model/pi_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/ppo_agent/ppo2_model/pi_3/add_grad/Shape4gradients/ppo_agent/ppo2_model/pi_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0gradients/ppo_agent/ppo2_model/pi_3/add_grad/SumSumgradients/AddN_5Bgradients/ppo_agent/ppo2_model/pi_3/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:	�
�
4gradients/ppo_agent/ppo2_model/pi_3/add_grad/ReshapeReshape0gradients/ppo_agent/ppo2_model/pi_3/add_grad/Sum2gradients/ppo_agent/ppo2_model/pi_3/add_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
2gradients/ppo_agent/ppo2_model/pi_3/add_grad/Sum_1Sumgradients/AddN_5Dgradients/ppo_agent/ppo2_model/pi_3/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
6gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1Reshape2gradients/ppo_agent/ppo2_model/pi_3/add_grad/Sum_14gradients/ppo_agent/ppo2_model/pi_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
=gradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/group_depsNoOp5^gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape7^gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1
�
Egradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependencyIdentity4gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape>^gradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape*
_output_shapes
:	�
�
Ggradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependency_1Identity6gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1>^gradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1*
_output_shapes
:
�
9gradients/ppo_agent/ppo2_model/strided_slice_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
Dgradients/ppo_agent/ppo2_model/strided_slice_1_grad/StridedSliceGradStridedSliceGrad9gradients/ppo_agent/ppo2_model/strided_slice_1_grad/Shape*ppo_agent/ppo2_model/strided_slice_1/stack,ppo_agent/ppo2_model/strided_slice_1/stack_1,ppo_agent/ppo2_model/strided_slice_1/stack_2gradients/AddN_4*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
_output_shapes
:	�
�
6gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMulMatMulEgradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependencyppo_agent/ppo2_model/pi/w/read*
T0*
transpose_a( *
transpose_b(*
_output_shapes
:	�@
�
8gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1MatMul&ppo_agent/ppo2_model/flatten_3/ReshapeEgradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:@
�
@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/group_depsNoOp7^gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul9^gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1
�
Hgradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/control_dependencyIdentity6gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMulA^gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul*
_output_shapes
:	�@
�
Jgradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/control_dependency_1Identity8gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1A^gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1*
_output_shapes

:@
�
2gradients/ppo_agent/ppo2_model/vf_1/add_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
~
4gradients/ppo_agent/ppo2_model/vf_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Bgradients/ppo_agent/ppo2_model/vf_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients/ppo_agent/ppo2_model/vf_1/add_grad/Shape4gradients/ppo_agent/ppo2_model/vf_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
0gradients/ppo_agent/ppo2_model/vf_1/add_grad/SumSumDgradients/ppo_agent/ppo2_model/strided_slice_1_grad/StridedSliceGradBgradients/ppo_agent/ppo2_model/vf_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:�
�
4gradients/ppo_agent/ppo2_model/vf_1/add_grad/ReshapeReshape0gradients/ppo_agent/ppo2_model/vf_1/add_grad/Sum2gradients/ppo_agent/ppo2_model/vf_1/add_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�
�
2gradients/ppo_agent/ppo2_model/vf_1/add_grad/Sum_1SumDgradients/ppo_agent/ppo2_model/strided_slice_1_grad/StridedSliceGradDgradients/ppo_agent/ppo2_model/vf_1/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
�
6gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1Reshape2gradients/ppo_agent/ppo2_model/vf_1/add_grad/Sum_14gradients/ppo_agent/ppo2_model/vf_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
=gradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/group_depsNoOp5^gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape7^gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1
�
Egradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependencyIdentity4gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape>^gradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape*
_output_shapes
:	�
�
Ggradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependency_1Identity6gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1>^gradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1*
_output_shapes
:
�
;gradients/ppo_agent/ppo2_model/flatten_3/Reshape_grad/ShapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
�
=gradients/ppo_agent/ppo2_model/flatten_3/Reshape_grad/ReshapeReshapeHgradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/control_dependency;gradients/ppo_agent/ppo2_model/flatten_3/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�@
�
6gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMulMatMulEgradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependencyppo_agent/ppo2_model/vf/w/read*
T0*
transpose_a( *
transpose_b(*
_output_shapes
:	�@
�
8gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1MatMul&ppo_agent/ppo2_model/flatten_2/ReshapeEgradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:@
�
@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/group_depsNoOp7^gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul9^gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1
�
Hgradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/control_dependencyIdentity6gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMulA^gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul*
_output_shapes
:	�@
�
Jgradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/control_dependency_1Identity8gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1A^gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1*
_output_shapes

:@
�
;gradients/ppo_agent/ppo2_model/flatten_2/Reshape_grad/ShapeConst*
valueB"   @   *
dtype0*
_output_shapes
:
�
=gradients/ppo_agent/ppo2_model/flatten_2/Reshape_grad/ReshapeReshapeHgradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/control_dependency;gradients/ppo_agent/ppo2_model/flatten_2/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes
:	�@
�
gradients/AddN_6AddN=gradients/ppo_agent/ppo2_model/flatten_3/Reshape_grad/Reshape=gradients/ppo_agent/ppo2_model/flatten_2/Reshape_grad/Reshape*
T0*P
_classF
DBloc:@gradients/ppo_agent/ppo2_model/flatten_3/Reshape_grad/Reshape*
N*
_output_shapes
:	�@
�
Hgradients/ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu_grad/LeakyReluGradLeakyReluGradgradients/AddN_6)ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd*
T0*
alpha%��L>*
_output_shapes
:	�@
�
Dgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradHgradients/ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
Igradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/group_depsNoOpE^gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGradI^gradients/ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu_grad/LeakyReluGrad
�
Qgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependencyIdentityHgradients/ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu_grad/LeakyReluGradJ^gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
:	�@
�
Sgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependency_1IdentityDgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGradJ^gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
>gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMulMatMulQgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependency+ppo_agent/ppo2_model/pi/dense_2/kernel/read*
transpose_a( *
transpose_b(*
T0*
_output_shapes
:	�@
�
@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1MatMul+ppo_agent/ppo2_model/pi_2/dense_1/LeakyReluQgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:@@
�
Hgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/group_depsNoOp?^gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMulA^gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1
�
Pgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/control_dependencyIdentity>gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMulI^gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul*
_output_shapes
:	�@
�
Rgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/control_dependency_1Identity@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1I^gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@@
�
Hgradients/ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradPgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/control_dependency)ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd*
T0*
alpha%��L>*
_output_shapes
:	�@
�
Dgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradHgradients/ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
Igradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/group_depsNoOpE^gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGradI^gradients/ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu_grad/LeakyReluGrad
�
Qgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependencyIdentityHgradients/ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu_grad/LeakyReluGradJ^gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
:	�@
�
Sgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependency_1IdentityDgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGradJ^gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
>gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMulMatMulQgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependency+ppo_agent/ppo2_model/pi/dense_1/kernel/read*
T0*
transpose_a( *
transpose_b(*
_output_shapes
:	�@
�
@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1MatMul)ppo_agent/ppo2_model/pi_2/dense/LeakyReluQgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:@@
�
Hgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/group_depsNoOp?^gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMulA^gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1
�
Pgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/control_dependencyIdentity>gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMulI^gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul*
_output_shapes
:	�@
�
Rgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/control_dependency_1Identity@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1I^gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
�
Fgradients/ppo_agent/ppo2_model/pi_2/dense/LeakyRelu_grad/LeakyReluGradLeakyReluGradPgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/control_dependency'ppo_agent/ppo2_model/pi_2/dense/BiasAdd*
T0*
alpha%��L>*
_output_shapes
:	�@
�
Bgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGradBiasAddGradFgradients/ppo_agent/ppo2_model/pi_2/dense/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
T0*
_output_shapes
:@
�
Ggradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/group_depsNoOpC^gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGradG^gradients/ppo_agent/ppo2_model/pi_2/dense/LeakyRelu_grad/LeakyReluGrad
�
Ogradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependencyIdentityFgradients/ppo_agent/ppo2_model/pi_2/dense/LeakyRelu_grad/LeakyReluGradH^gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/LeakyRelu_grad/LeakyReluGrad*
_output_shapes
:	�@
�
Qgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependency_1IdentityBgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGradH^gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
<gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMulMatMulOgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependency)ppo_agent/ppo2_model/pi/dense/kernel/read*
T0*
transpose_a( *
transpose_b(* 
_output_shapes
:
��
�
>gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1MatMul)ppo_agent/ppo2_model/pi_2/flatten/ReshapeOgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes
:	�@
�
Fgradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/group_depsNoOp=^gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul?^gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1
�
Ngradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/control_dependencyIdentity<gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMulG^gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul* 
_output_shapes
:
��
�
Pgradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/control_dependency_1Identity>gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1G^gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�@
�
>gradients/ppo_agent/ppo2_model/pi_2/flatten/Reshape_grad/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
�
@gradients/ppo_agent/ppo2_model/pi_2/flatten/Reshape_grad/ReshapeReshapeNgradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/control_dependency>gradients/ppo_agent/ppo2_model/pi_2/flatten/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:�
�
Ggradients/ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu_grad/LeakyReluGradLeakyReluGrad@gradients/ppo_agent/ppo2_model/pi_2/flatten/Reshape_grad/Reshape(ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd*
T0*
alpha%��L>*'
_output_shapes
:�
�
Cgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGradBiasAddGradGgradients/ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
�
Hgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/group_depsNoOpD^gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGradH^gradients/ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu_grad/LeakyReluGrad
�
Pgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependencyIdentityGgradients/ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu_grad/LeakyReluGradI^gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/LeakyRelu_grad/LeakyReluGrad*'
_output_shapes
:�
�
Rgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGradI^gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
=gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/ShapeNShapeN*ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu*ppo_agent/ppo2_model/pi/conv_1/kernel/read*
N*
T0*
out_type0* 
_output_shapes
::
�
Jgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/ShapeN*ppo_agent/ppo2_model/pi/conv_1/kernel/readPgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
	dilations
*'
_output_shapes
:�
�
Kgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter*ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu?gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/ShapeN:1Pgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
Ggradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/group_depsNoOpL^gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilterK^gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropInput
�
Ogradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/control_dependencyIdentityJgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropInputH^gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
:�
�
Qgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/control_dependency_1IdentityKgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilterH^gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
Ggradients/ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu_grad/LeakyReluGradLeakyReluGradOgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/control_dependency(ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd*
T0*
alpha%��L>*'
_output_shapes
:�
�
Cgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGradBiasAddGradGgradients/ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
Hgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/group_depsNoOpD^gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGradH^gradients/ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu_grad/LeakyReluGrad
�
Pgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependencyIdentityGgradients/ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu_grad/LeakyReluGradI^gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/LeakyRelu_grad/LeakyReluGrad*'
_output_shapes
:�
�
Rgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGradI^gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
=gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/ShapeNShapeN0ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu*ppo_agent/ppo2_model/pi/conv_0/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
Jgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/ShapeN*ppo_agent/ppo2_model/pi/conv_0/kernel/readPgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�
�
Kgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter0ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu?gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/ShapeN:1Pgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*&
_output_shapes
:
�
Ggradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/group_depsNoOpL^gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilterK^gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropInput
�
Ogradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/control_dependencyIdentityJgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropInputH^gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
:�
�
Qgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/control_dependency_1IdentityKgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilterH^gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
Mgradients/ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu_grad/LeakyReluGradLeakyReluGradOgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/control_dependency.ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd*
T0*
alpha%��L>*'
_output_shapes
:�
�
Igradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGradBiasAddGradMgradients/ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
Ngradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/group_depsNoOpJ^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGradN^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu_grad/LeakyReluGrad
�
Vgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependencyIdentityMgradients/ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu_grad/LeakyReluGradO^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/LeakyRelu_grad/LeakyReluGrad*'
_output_shapes
:�
�
Xgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGradO^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
Cgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/ShapeNShapeNppo_agent/ppo2_model/Ob_10ppo_agent/ppo2_model/pi/conv_initial/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
�
Pgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/ShapeN0ppo_agent/ppo2_model/pi/conv_initial/kernel/readVgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�
�
Qgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterppo_agent/ppo2_model/Ob_1Egradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/ShapeN:1Vgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*&
_output_shapes
:
�
Mgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/group_depsNoOpR^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilterQ^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropInput
�
Ugradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/control_dependencyIdentityPgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropInputN^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
:�
�
Wgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/control_dependency_1IdentityQgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilterN^gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
global_norm/L2LossL2LossWgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/control_dependency_1*
T0*d
_classZ
XVloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
global_norm/L2Loss_1L2LossXgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependency_1*
T0*\
_classR
PNloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_2L2LossQgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
global_norm/L2Loss_3L2LossRgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependency_1*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_4L2LossQgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilter*
_output_shapes
: 
�
global_norm/L2Loss_5L2LossRgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependency_1*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_6L2LossPgradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_7L2LossQgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*U
_classK
IGloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_8L2LossRgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_9L2LossSgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_10L2LossRgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_11L2LossSgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
�
global_norm/L2Loss_12L2LossJgradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_13L2LossGgradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependency_1*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1*
_output_shapes
: 
�
global_norm/L2Loss_14L2LossJgradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1*
_output_shapes
: 
�
global_norm/L2Loss_15L2LossGgradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependency_1*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1*
_output_shapes
: 
�
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5global_norm/L2Loss_6global_norm/L2Loss_7global_norm/L2Loss_8global_norm/L2Loss_9global_norm/L2Loss_10global_norm/L2Loss_11global_norm/L2Loss_12global_norm/L2Loss_13global_norm/L2Loss_14global_norm/L2Loss_15*
T0*

axis *
N*
_output_shapes
:
[
global_norm/ConstConst*
valueB: *
dtype0*
_output_shapes
:
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
X
global_norm/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
T0*
_output_shapes
: 
Q
global_norm/global_normSqrtglobal_norm/mul*
T0*
_output_shapes
: 
�
VerifyFinite/CheckNumericsCheckNumericsglobal_norm/global_norm**
messageFound Inf or NaN global norm.*
T0**
_class 
loc:@global_norm/global_norm*
_output_shapes
: 
�
VerifyFinite/control_dependencyIdentityglobal_norm/global_norm^VerifyFinite/CheckNumerics*
T0**
_class 
loc:@global_norm/global_norm*
_output_shapes
: 
b
clip_by_global_norm/truediv/xConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xVerifyFinite/control_dependency*
T0*
_output_shapes
: 
^
clip_by_global_norm/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
d
clip_by_global_norm/truediv_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 
�
clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
^
clip_by_global_norm/mul/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
T0*
_output_shapes
: 
�
clip_by_global_norm/mul_1MulWgradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*d
_classZ
XVloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0*d
_classZ
XVloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
clip_by_global_norm/mul_2MulXgradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*\
_classR
PNloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0*\
_classR
PNloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_initial/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
clip_by_global_norm/mul_3MulQgradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
clip_by_global_norm/mul_4MulRgradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_0/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
clip_by_global_norm/mul_5MulQgradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
T0*^
_classT
RPloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
clip_by_global_norm/mul_6MulRgradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
T0*V
_classL
JHloc:@gradients/ppo_agent/ppo2_model/pi_2/conv_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
clip_by_global_norm/mul_7MulPgradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�@
�
*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7*
T0*Q
_classG
ECloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/MatMul_grad/MatMul_1*
_output_shapes
:	�@
�
clip_by_global_norm/mul_8MulQgradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*U
_classK
IGloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8*
T0*U
_classK
IGloc:@gradients/ppo_agent/ppo2_model/pi_2/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
clip_by_global_norm/mul_9MulRgradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
�
*clip_by_global_norm/clip_by_global_norm/_8Identityclip_by_global_norm/mul_9*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
�
clip_by_global_norm/mul_10MulSgradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
*clip_by_global_norm/clip_by_global_norm/_9Identityclip_by_global_norm/mul_10*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
clip_by_global_norm/mul_11MulRgradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@@
�
+clip_by_global_norm/clip_by_global_norm/_10Identityclip_by_global_norm/mul_11*
T0*S
_classI
GEloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@@
�
clip_by_global_norm/mul_12MulSgradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
+clip_by_global_norm/clip_by_global_norm/_11Identityclip_by_global_norm/mul_12*
T0*W
_classM
KIloc:@gradients/ppo_agent/ppo2_model/pi_2/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
clip_by_global_norm/mul_13MulJgradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1*
_output_shapes

:@
�
+clip_by_global_norm/clip_by_global_norm/_12Identityclip_by_global_norm/mul_13*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/pi_3/MatMul_grad/MatMul_1*
_output_shapes

:@
�
clip_by_global_norm/mul_14MulGgradients/ppo_agent/ppo2_model/pi_3/add_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1*
_output_shapes
:
�
+clip_by_global_norm/clip_by_global_norm/_13Identityclip_by_global_norm/mul_14*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/pi_3/add_grad/Reshape_1*
_output_shapes
:
�
clip_by_global_norm/mul_15MulJgradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1*
_output_shapes

:@
�
+clip_by_global_norm/clip_by_global_norm/_14Identityclip_by_global_norm/mul_15*
T0*K
_classA
?=loc:@gradients/ppo_agent/ppo2_model/vf_1/MatMul_grad/MatMul_1*
_output_shapes

:@
�
clip_by_global_norm/mul_16MulGgradients/ppo_agent/ppo2_model/vf_1/add_grad/tuple/control_dependency_1clip_by_global_norm/mul*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1*
_output_shapes
:
�
+clip_by_global_norm/clip_by_global_norm/_15Identityclip_by_global_norm/mul_16*
T0*I
_class?
=;loc:@gradients/ppo_agent/ppo2_model/vf_1/add_grad/Reshape_1*
_output_shapes
:
�
beta1_power/initial_valueConst*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
dtype0*
	container *
shape: *
shared_name *
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
x
beta1_power/readIdentitybeta1_power*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
beta2_power/initial_valueConst*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
shape: *
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
dtype0*
	container *
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
x
beta2_power/readIdentitybeta2_power*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
Rppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"            *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
dtype0*
_output_shapes
:
�
Hppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
dtype0*
_output_shapes
: 
�
Bppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zerosFillRppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros/shape_as_tensorHppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam
VariableV2*
shared_name *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
dtype0*
	container *
shape:*&
_output_shapes
:
�
7ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/AssignAssign0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamBppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
5ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/readIdentity0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
Tppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
dtype0*
_output_shapes
:
�
Jppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
dtype0*
_output_shapes
: 
�
Dppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zerosFillTppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros/shape_as_tensorJppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1
VariableV2*
shared_name *>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
dtype0*
	container *
shape:*&
_output_shapes
:
�
9ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/AssignAssign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1Dppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
7ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/readIdentity2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
@ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/Initializer/zerosConst*
valueB*    *<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
dtype0*
_output_shapes
:
�
.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam
VariableV2*
dtype0*
	container *
shape:*
shared_name *<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
5ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/AssignAssign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam@ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
3ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/readIdentity.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
Bppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1
VariableV2*
dtype0*
	container *
shape:*
shared_name *<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
7ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/AssignAssign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1Bppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/Initializer/zeros*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
5ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/readIdentity0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
Lppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"            *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
dtype0*
_output_shapes
:
�
Bppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
dtype0*
_output_shapes
: 
�
<ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zerosFillLppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros/shape_as_tensorBppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam
VariableV2*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
dtype0*
	container *
shape:*
shared_name *&
_output_shapes
:
�
1ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/AssignAssign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam<ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
/ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/readIdentity*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
Nppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*%
valueB"            *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
_output_shapes
:
�
Dppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
_output_shapes
: 
�
>ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zerosFillNppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros/shape_as_tensorDppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1
VariableV2*
shared_name *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
dtype0*
	container *
shape:*&
_output_shapes
:
�
3ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/AssignAssign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1>ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
1ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/readIdentity,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
:ppo_agent/ppo2_model/pi/conv_0/bias/Adam/Initializer/zerosConst*
valueB*    *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
dtype0*
_output_shapes
:
�
(ppo_agent/ppo2_model/pi/conv_0/bias/Adam
VariableV2*
dtype0*
	container *
shape:*
shared_name *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
/ppo_agent/ppo2_model/pi/conv_0/bias/Adam/AssignAssign(ppo_agent/ppo2_model/pi/conv_0/bias/Adam:ppo_agent/ppo2_model/pi/conv_0/bias/Adam/Initializer/zeros*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
-ppo_agent/ppo2_model/pi/conv_0/bias/Adam/readIdentity(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
<ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1
VariableV2*
shared_name *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
dtype0*
	container *
shape:*
_output_shapes
:
�
1ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/AssignAssign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1<ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
/ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/readIdentity*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
Lppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*%
valueB"            *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
_output_shapes
:
�
Bppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
dtype0*
_output_shapes
: 
�
<ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zerosFillLppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros/shape_as_tensorBppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam
VariableV2*
shared_name *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
dtype0*
	container *
shape:*&
_output_shapes
:
�
1ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/AssignAssign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam<ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
/ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/readIdentity*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
Nppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"            *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
dtype0*
_output_shapes
:
�
Dppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
dtype0*
_output_shapes
: 
�
>ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zerosFillNppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorDppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1
VariableV2*
shared_name *8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
dtype0*
	container *
shape:*&
_output_shapes
:
�
3ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/AssignAssign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1>ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
1ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/readIdentity,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
:ppo_agent/ppo2_model/pi/conv_1/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
(ppo_agent/ppo2_model/pi/conv_1/bias/Adam
VariableV2*
shape:*
shared_name *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
dtype0*
	container *
_output_shapes
:
�
/ppo_agent/ppo2_model/pi/conv_1/bias/Adam/AssignAssign(ppo_agent/ppo2_model/pi/conv_1/bias/Adam:ppo_agent/ppo2_model/pi/conv_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
-ppo_agent/ppo2_model/pi/conv_1/bias/Adam/readIdentity(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
<ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/Initializer/zerosConst*
valueB*    *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
dtype0*
_output_shapes
:
�
*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1
VariableV2*
shared_name *6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
dtype0*
	container *
shape:*
_output_shapes
:
�
1ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/AssignAssign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1<ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
/ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/readIdentity*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
Kppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"�  @   *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
dtype0*
_output_shapes
:
�
Appo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
: 
�
;ppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zerosFillKppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorAppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
)ppo_agent/ppo2_model/pi/dense/kernel/Adam
VariableV2*
shape:	�@*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
dtype0*
	container *
_output_shapes
:	�@
�
0ppo_agent/ppo2_model/pi/dense/kernel/Adam/AssignAssign)ppo_agent/ppo2_model/pi/dense/kernel/Adam;ppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
.ppo_agent/ppo2_model/pi/dense/kernel/Adam/readIdentity)ppo_agent/ppo2_model/pi/dense/kernel/Adam*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
Mppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"�  @   *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
dtype0*
_output_shapes
:
�
Cppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
dtype0*
_output_shapes
: 
�
=ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zerosFillMppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorCppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1
VariableV2*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
dtype0*
	container *
shape:	�@*
shared_name *
_output_shapes
:	�@
�
2ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/AssignAssign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1=ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
0ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/readIdentity+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
9ppo_agent/ppo2_model/pi/dense/bias/Adam/Initializer/zerosConst*
valueB@*    *5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
dtype0*
_output_shapes
:@
�
'ppo_agent/ppo2_model/pi/dense/bias/Adam
VariableV2*
shape:@*
shared_name *5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
dtype0*
	container *
_output_shapes
:@
�
.ppo_agent/ppo2_model/pi/dense/bias/Adam/AssignAssign'ppo_agent/ppo2_model/pi/dense/bias/Adam9ppo_agent/ppo2_model/pi/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
,ppo_agent/ppo2_model/pi/dense/bias/Adam/readIdentity'ppo_agent/ppo2_model/pi/dense/bias/Adam*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
;ppo_agent/ppo2_model/pi/dense/bias/Adam_1/Initializer/zerosConst*
valueB@*    *5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
dtype0*
_output_shapes
:@
�
)ppo_agent/ppo2_model/pi/dense/bias/Adam_1
VariableV2*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
dtype0*
	container *
shape:@*
shared_name *
_output_shapes
:@
�
0ppo_agent/ppo2_model/pi/dense/bias/Adam_1/AssignAssign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1;ppo_agent/ppo2_model/pi/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
.ppo_agent/ppo2_model/pi/dense/bias/Adam_1/readIdentity)ppo_agent/ppo2_model/pi/dense/bias/Adam_1*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
Mppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
dtype0*
_output_shapes
:
�
Cppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes
: 
�
=ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zerosFillMppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorCppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam
VariableV2*
shape
:@@*
shared_name *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
dtype0*
	container *
_output_shapes

:@@
�
2ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/AssignAssign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam=ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
0ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/readIdentity+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
Oppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
dtype0*
_output_shapes
:
�
Eppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
�
?ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zerosFillOppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorEppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1
VariableV2*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
dtype0*
	container *
shape
:@@*
shared_name *
_output_shapes

:@@
�
4ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/AssignAssign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1?ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
2ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/readIdentity-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
;ppo_agent/ppo2_model/pi/dense_1/bias/Adam/Initializer/zerosConst*
valueB@*    *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
dtype0*
_output_shapes
:@
�
)ppo_agent/ppo2_model/pi/dense_1/bias/Adam
VariableV2*
dtype0*
	container *
shape:@*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
0ppo_agent/ppo2_model/pi/dense_1/bias/Adam/AssignAssign)ppo_agent/ppo2_model/pi/dense_1/bias/Adam;ppo_agent/ppo2_model/pi/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
.ppo_agent/ppo2_model/pi/dense_1/bias/Adam/readIdentity)ppo_agent/ppo2_model/pi/dense_1/bias/Adam*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
=ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB@*    *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
dtype0*
_output_shapes
:@
�
+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1
VariableV2*
dtype0*
	container *
shape:@*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
2ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/AssignAssign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1=ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/Initializer/zeros*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
0ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/readIdentity+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
Mppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"@   @   *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes
:
�
Cppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
dtype0*
_output_shapes
: 
�
=ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zerosFillMppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorCppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam
VariableV2*
dtype0*
	container *
shape
:@@*
shared_name *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
2ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/AssignAssign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam=ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
0ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/readIdentity+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
Oppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"@   @   *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes
:
�
Eppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
dtype0*
_output_shapes
: 
�
?ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zerosFillOppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorEppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1
VariableV2*
shared_name *9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
dtype0*
	container *
shape
:@@*
_output_shapes

:@@
�
4ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/AssignAssign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1?ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
2ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/readIdentity-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
;ppo_agent/ppo2_model/pi/dense_2/bias/Adam/Initializer/zerosConst*
valueB@*    *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
dtype0*
_output_shapes
:@
�
)ppo_agent/ppo2_model/pi/dense_2/bias/Adam
VariableV2*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
dtype0*
	container *
shape:@*
shared_name *
_output_shapes
:@
�
0ppo_agent/ppo2_model/pi/dense_2/bias/Adam/AssignAssign)ppo_agent/ppo2_model/pi/dense_2/bias/Adam;ppo_agent/ppo2_model/pi/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
.ppo_agent/ppo2_model/pi/dense_2/bias/Adam/readIdentity)ppo_agent/ppo2_model/pi/dense_2/bias/Adam*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
=ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/Initializer/zerosConst*
valueB@*    *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
dtype0*
_output_shapes
:@
�
+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1
VariableV2*
dtype0*
	container *
shape:@*
shared_name *7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
2ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/AssignAssign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1=ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
0ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/readIdentity+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
0ppo_agent/ppo2_model/pi/w/Adam/Initializer/zerosConst*
dtype0*
valueB@*    *,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
ppo_agent/ppo2_model/pi/w/Adam
VariableV2*
shape
:@*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
dtype0*
	container *
_output_shapes

:@
�
%ppo_agent/ppo2_model/pi/w/Adam/AssignAssignppo_agent/ppo2_model/pi/w/Adam0ppo_agent/ppo2_model/pi/w/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
#ppo_agent/ppo2_model/pi/w/Adam/readIdentityppo_agent/ppo2_model/pi/w/Adam*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
2ppo_agent/ppo2_model/pi/w/Adam_1/Initializer/zerosConst*
valueB@*    *,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
dtype0*
_output_shapes

:@
�
 ppo_agent/ppo2_model/pi/w/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
dtype0*
	container *
shape
:@*
_output_shapes

:@
�
'ppo_agent/ppo2_model/pi/w/Adam_1/AssignAssign ppo_agent/ppo2_model/pi/w/Adam_12ppo_agent/ppo2_model/pi/w/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
%ppo_agent/ppo2_model/pi/w/Adam_1/readIdentity ppo_agent/ppo2_model/pi/w/Adam_1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
0ppo_agent/ppo2_model/pi/b/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
dtype0*
_output_shapes
:
�
ppo_agent/ppo2_model/pi/b/Adam
VariableV2*
shape:*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
dtype0*
	container *
_output_shapes
:
�
%ppo_agent/ppo2_model/pi/b/Adam/AssignAssignppo_agent/ppo2_model/pi/b/Adam0ppo_agent/ppo2_model/pi/b/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
#ppo_agent/ppo2_model/pi/b/Adam/readIdentityppo_agent/ppo2_model/pi/b/Adam*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
2ppo_agent/ppo2_model/pi/b/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
 ppo_agent/ppo2_model/pi/b/Adam_1
VariableV2*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
dtype0*
	container *
shape:*
shared_name *
_output_shapes
:
�
'ppo_agent/ppo2_model/pi/b/Adam_1/AssignAssign ppo_agent/ppo2_model/pi/b/Adam_12ppo_agent/ppo2_model/pi/b/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
%ppo_agent/ppo2_model/pi/b/Adam_1/readIdentity ppo_agent/ppo2_model/pi/b/Adam_1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
0ppo_agent/ppo2_model/vf/w/Adam/Initializer/zerosConst*
valueB@*    *,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
dtype0*
_output_shapes

:@
�
ppo_agent/ppo2_model/vf/w/Adam
VariableV2*
dtype0*
	container *
shape
:@*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
%ppo_agent/ppo2_model/vf/w/Adam/AssignAssignppo_agent/ppo2_model/vf/w/Adam0ppo_agent/ppo2_model/vf/w/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
#ppo_agent/ppo2_model/vf/w/Adam/readIdentityppo_agent/ppo2_model/vf/w/Adam*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
2ppo_agent/ppo2_model/vf/w/Adam_1/Initializer/zerosConst*
dtype0*
valueB@*    *,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
 ppo_agent/ppo2_model/vf/w/Adam_1
VariableV2*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
dtype0*
	container *
shape
:@*
shared_name *
_output_shapes

:@
�
'ppo_agent/ppo2_model/vf/w/Adam_1/AssignAssign ppo_agent/ppo2_model/vf/w/Adam_12ppo_agent/ppo2_model/vf/w/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
%ppo_agent/ppo2_model/vf/w/Adam_1/readIdentity ppo_agent/ppo2_model/vf/w/Adam_1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
0ppo_agent/ppo2_model/vf/b/Adam/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
dtype0*
_output_shapes
:
�
ppo_agent/ppo2_model/vf/b/Adam
VariableV2*
dtype0*
	container *
shape:*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
%ppo_agent/ppo2_model/vf/b/Adam/AssignAssignppo_agent/ppo2_model/vf/b/Adam0ppo_agent/ppo2_model/vf/b/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
#ppo_agent/ppo2_model/vf/b/Adam/readIdentityppo_agent/ppo2_model/vf/b/Adam*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
2ppo_agent/ppo2_model/vf/b/Adam_1/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
dtype0*
_output_shapes
:
�
 ppo_agent/ppo2_model/vf/b/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
dtype0*
	container *
shape:*
_output_shapes
:
�
'ppo_agent/ppo2_model/vf/b/Adam_1/AssignAssign ppo_agent/ppo2_model/vf/b/Adam_12ppo_agent/ppo2_model/vf/b/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
%ppo_agent/ppo2_model/vf/b/Adam_1/readIdentity ppo_agent/ppo2_model/vf/b/Adam_1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
O

Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *��'7*
dtype0*
_output_shapes
: 
�
AAdam/update_ppo_agent/ppo2_model/pi/conv_initial/kernel/ApplyAdam	ApplyAdam+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
use_locking( *
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
use_nesterov( *&
_output_shapes
:
�
?Adam/update_ppo_agent/ppo2_model/pi/conv_initial/bias/ApplyAdam	ApplyAdam)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
use_locking( *
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
use_nesterov( *
_output_shapes
:
�
;Adam/update_ppo_agent/ppo2_model/pi/conv_0/kernel/ApplyAdam	ApplyAdam%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
use_nesterov( *
use_locking( *
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
9Adam/update_ppo_agent/ppo2_model/pi/conv_0/bias/ApplyAdam	ApplyAdam#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
use_nesterov( *
use_locking( *
_output_shapes
:
�
;Adam/update_ppo_agent/ppo2_model/pi/conv_1/kernel/ApplyAdam	ApplyAdam%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
use_nesterov( *
use_locking( *
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
9Adam/update_ppo_agent/ppo2_model/pi/conv_1/bias/ApplyAdam	ApplyAdam#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*
use_locking( *
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
use_nesterov( *
_output_shapes
:
�
:Adam/update_ppo_agent/ppo2_model/pi/dense/kernel/ApplyAdam	ApplyAdam$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_6*
use_locking( *
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
use_nesterov( *
_output_shapes
:	�@
�
8Adam/update_ppo_agent/ppo2_model/pi/dense/bias/ApplyAdam	ApplyAdam"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_7*
use_locking( *
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
use_nesterov( *
_output_shapes
:@
�
<Adam/update_ppo_agent/ppo2_model/pi/dense_1/kernel/ApplyAdam	ApplyAdam&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_8*
use_nesterov( *
use_locking( *
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
:Adam/update_ppo_agent/ppo2_model/pi/dense_1/bias/ApplyAdam	ApplyAdam$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_9*
use_nesterov( *
use_locking( *
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
<Adam/update_ppo_agent/ppo2_model/pi/dense_2/kernel/ApplyAdam	ApplyAdam&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_10*
use_locking( *
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
use_nesterov( *
_output_shapes

:@@
�
:Adam/update_ppo_agent/ppo2_model/pi/dense_2/bias/ApplyAdam	ApplyAdam$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_11*
use_locking( *
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
use_nesterov( *
_output_shapes
:@
�
/Adam/update_ppo_agent/ppo2_model/pi/w/ApplyAdam	ApplyAdamppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_12*
use_nesterov( *
use_locking( *
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
/Adam/update_ppo_agent/ppo2_model/pi/b/ApplyAdam	ApplyAdamppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_13*
use_locking( *
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
use_nesterov( *
_output_shapes
:
�
/Adam/update_ppo_agent/ppo2_model/vf/w/ApplyAdam	ApplyAdamppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_14*
use_locking( *
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
use_nesterov( *
_output_shapes

:@
�
/Adam/update_ppo_agent/ppo2_model/vf/b/ApplyAdam	ApplyAdamppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1beta1_power/readbeta2_power/readPlaceholder_5
Adam/beta1
Adam/beta2Adam/epsilon+clip_by_global_norm/clip_by_global_norm/_15*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
use_nesterov( *
use_locking( *
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta10^Adam/update_ppo_agent/ppo2_model/pi/b/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_0/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_0/kernel/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_1/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_1/kernel/ApplyAdam@^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/bias/ApplyAdamB^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/kernel/ApplyAdam9^Adam/update_ppo_agent/ppo2_model/pi/dense/bias/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_1/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_1/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_2/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_2/kernel/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/pi/w/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/b/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/w/ApplyAdam*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking( *
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta20^Adam/update_ppo_agent/ppo2_model/pi/b/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_0/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_0/kernel/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_1/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_1/kernel/ApplyAdam@^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/bias/ApplyAdamB^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/kernel/ApplyAdam9^Adam/update_ppo_agent/ppo2_model/pi/dense/bias/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_1/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_1/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_2/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_2/kernel/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/pi/w/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/b/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/w/ApplyAdam*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
AdamNoOp^Adam/Assign^Adam/Assign_10^Adam/update_ppo_agent/ppo2_model/pi/b/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_0/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_0/kernel/ApplyAdam:^Adam/update_ppo_agent/ppo2_model/pi/conv_1/bias/ApplyAdam<^Adam/update_ppo_agent/ppo2_model/pi/conv_1/kernel/ApplyAdam@^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/bias/ApplyAdamB^Adam/update_ppo_agent/ppo2_model/pi/conv_initial/kernel/ApplyAdam9^Adam/update_ppo_agent/ppo2_model/pi/dense/bias/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_1/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_1/kernel/ApplyAdam;^Adam/update_ppo_agent/ppo2_model/pi/dense_2/bias/ApplyAdam=^Adam/update_ppo_agent/ppo2_model/pi/dense_2/kernel/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/pi/w/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/b/ApplyAdam0^Adam/update_ppo_agent/ppo2_model/vf/w/ApplyAdam
�
initNoOp^beta1_power/Assign^beta2_power/Assign&^ppo_agent/ppo2_model/pi/b/Adam/Assign(^ppo_agent/ppo2_model/pi/b/Adam_1/Assign!^ppo_agent/ppo2_model/pi/b/Assign0^ppo_agent/ppo2_model/pi/conv_0/bias/Adam/Assign2^ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/Assign+^ppo_agent/ppo2_model/pi/conv_0/bias/Assign2^ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Assign4^ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Assign-^ppo_agent/ppo2_model/pi/conv_0/kernel/Assign0^ppo_agent/ppo2_model/pi/conv_1/bias/Adam/Assign2^ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/Assign+^ppo_agent/ppo2_model/pi/conv_1/bias/Assign2^ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Assign4^ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Assign-^ppo_agent/ppo2_model/pi/conv_1/kernel/Assign6^ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/Assign8^ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/Assign1^ppo_agent/ppo2_model/pi/conv_initial/bias/Assign8^ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Assign:^ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Assign3^ppo_agent/ppo2_model/pi/conv_initial/kernel/Assign/^ppo_agent/ppo2_model/pi/dense/bias/Adam/Assign1^ppo_agent/ppo2_model/pi/dense/bias/Adam_1/Assign*^ppo_agent/ppo2_model/pi/dense/bias/Assign1^ppo_agent/ppo2_model/pi/dense/kernel/Adam/Assign3^ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Assign,^ppo_agent/ppo2_model/pi/dense/kernel/Assign1^ppo_agent/ppo2_model/pi/dense_1/bias/Adam/Assign3^ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/Assign,^ppo_agent/ppo2_model/pi/dense_1/bias/Assign3^ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Assign5^ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Assign.^ppo_agent/ppo2_model/pi/dense_1/kernel/Assign1^ppo_agent/ppo2_model/pi/dense_2/bias/Adam/Assign3^ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/Assign,^ppo_agent/ppo2_model/pi/dense_2/bias/Assign3^ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Assign5^ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Assign.^ppo_agent/ppo2_model/pi/dense_2/kernel/Assign&^ppo_agent/ppo2_model/pi/w/Adam/Assign(^ppo_agent/ppo2_model/pi/w/Adam_1/Assign!^ppo_agent/ppo2_model/pi/w/Assign&^ppo_agent/ppo2_model/vf/b/Adam/Assign(^ppo_agent/ppo2_model/vf/b/Adam_1/Assign!^ppo_agent/ppo2_model/vf/b/Assign&^ppo_agent/ppo2_model/vf/w/Adam/Assign(^ppo_agent/ppo2_model/vf/w/Adam_1/Assign!^ppo_agent/ppo2_model/vf/w/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: *
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: *
_output_shapes
: 
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_2e4e525fcd3e403aa9eb2081ad1e77be/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save/SaveV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
save/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save/AssignAssignbeta1_powersave/RestoreV2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save/Assign_2Assignppo_agent/ppo2_model/pi/bsave/RestoreV2:2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave/RestoreV2:3*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save/RestoreV2:4*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save/RestoreV2:7*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave/RestoreV2:11*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save/RestoreV2:13*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave/RestoreV2:14*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave/RestoreV2:20*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave/RestoreV2:23*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave/RestoreV2:29*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave/RestoreV2:30*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save/RestoreV2:31*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave/RestoreV2:32*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave/RestoreV2:33*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave/RestoreV2:38*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave/RestoreV2:39*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save/RestoreV2:40*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save/Assign_41Assignppo_agent/ppo2_model/pi/wsave/RestoreV2:41*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save/Assign_44Assignppo_agent/ppo2_model/vf/bsave/RestoreV2:44*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save/RestoreV2:46*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save/Assign_47Assignppo_agent/ppo2_model/vf/wsave/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave/RestoreV2:48*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_c6e48fd817bc484985623943c7ff8127/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
�
save_1/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_1/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
�
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
�
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
�
save_1/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_1Assignbeta2_powersave_1/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_1/Assign_2Assignppo_agent/ppo2_model/pi/bsave_1/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_1/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_1/RestoreV2:4*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_1/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_1/RestoreV2:5*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_1/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_1/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_1/RestoreV2:7*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_1/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_1/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_1/RestoreV2:9*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_1/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_1/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_1/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_1/RestoreV2:12*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_1/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_1/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_1/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_1/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_1/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_1/RestoreV2:17*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_1/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_1/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_1/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_1/RestoreV2:20*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_1/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_1/RestoreV2:21*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_1/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_1/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_1/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_1/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_1/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_1/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_1/RestoreV2:26*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_1/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_1/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_1/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_1/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_1/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_1/RestoreV2:29*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_1/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_1/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_1/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_1/RestoreV2:32*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_1/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_1/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_1/RestoreV2:34*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_1/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_1/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_1/RestoreV2:36*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_1/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_1/RestoreV2:37*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_1/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_1/RestoreV2:38*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_1/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_1/RestoreV2:39*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_1/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_1/RestoreV2:40*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_1/Assign_41Assignppo_agent/ppo2_model/pi/wsave_1/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_1/RestoreV2:42*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_1/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_1/RestoreV2:43*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_1/Assign_44Assignppo_agent/ppo2_model/vf/bsave_1/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_1/RestoreV2:45*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_1/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_1/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_1/Assign_47Assignppo_agent/ppo2_model/vf/wsave_1/RestoreV2:47*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_1/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_1/RestoreV2:48*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_1/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_1/RestoreV2:49*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_1faa4d61da1343f095bc67d494e414ba/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
�
save_2/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_2/SaveV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
�
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
N*
T0*

axis *
_output_shapes
:
�
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
�
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
�
save_2/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_2/RestoreV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_2/Assign_1Assignbeta2_powersave_2/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_2/Assign_2Assignppo_agent/ppo2_model/pi/bsave_2/RestoreV2:2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_2/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_2/RestoreV2:3*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_2/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_2/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_2/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_2/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_2/RestoreV2:6*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_2/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_2/RestoreV2:7*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_2/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_2/RestoreV2:8*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_2/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_2/RestoreV2:9*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_2/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_2/RestoreV2:10*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_2/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_2/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_2/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_2/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_2/RestoreV2:14*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_2/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_2/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_2/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_2/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_2/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_2/RestoreV2:17*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_2/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_2/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_2/RestoreV2:19*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_2/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_2/RestoreV2:20*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_2/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_2/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_2/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_2/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_2/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_2/RestoreV2:23*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_2/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_2/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_2/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_2/RestoreV2:26*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_2/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_2/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_2/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_2/RestoreV2:28*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_2/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_2/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_2/RestoreV2:30*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_2/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_2/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_2/RestoreV2:32*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_2/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_2/RestoreV2:33*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_2/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_2/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_2/RestoreV2:35*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_2/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_2/RestoreV2:36*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_2/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_2/RestoreV2:37*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_2/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_2/RestoreV2:38*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_2/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_2/RestoreV2:39*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_2/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_2/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_41Assignppo_agent/ppo2_model/pi/wsave_2/RestoreV2:41*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_2/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_2/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_2/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_2/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_44Assignppo_agent/ppo2_model/vf/bsave_2/RestoreV2:44*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_2/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_2/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_2/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_2/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_2/Assign_47Assignppo_agent/ppo2_model/vf/wsave_2/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_2/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_2/RestoreV2:49*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_45^save_2/Assign_46^save_2/Assign_47^save_2/Assign_48^save_2/Assign_49^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_81c8b9a1e5424ad4865106b764cddd07/part*
dtype0*
_output_shapes
: 
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_3/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
�
save_3/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_3/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 
�
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
�
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
�
save_3/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_3/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_1Assignbeta2_powersave_3/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_2Assignppo_agent/ppo2_model/pi/bsave_3/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_3/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_3/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_3/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_3/RestoreV2:4*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_3/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_3/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_3/RestoreV2:6*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_3/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_3/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_3/RestoreV2:8*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_3/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_3/RestoreV2:9*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_3/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_3/RestoreV2:10*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_3/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_3/RestoreV2:11*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_3/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_3/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_3/RestoreV2:13*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_3/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_3/RestoreV2:14*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_3/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_3/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_3/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_3/RestoreV2:16*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_3/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_3/RestoreV2:17*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_3/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_3/RestoreV2:18*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_3/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_3/RestoreV2:19*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_3/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_3/RestoreV2:20*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_3/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_3/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_3/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_3/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_3/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_3/RestoreV2:23*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_3/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_3/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_3/RestoreV2:25*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_3/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_3/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_3/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_3/RestoreV2:27*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_3/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_3/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_3/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_3/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_3/RestoreV2:30*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_3/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_3/RestoreV2:31*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_3/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_3/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_3/RestoreV2:33*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_3/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_3/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_3/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_3/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_3/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_3/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_3/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_3/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_41Assignppo_agent/ppo2_model/pi/wsave_3/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_3/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_3/RestoreV2:43*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_3/Assign_44Assignppo_agent/ppo2_model/vf/bsave_3/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_3/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_3/RestoreV2:45*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_3/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_3/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_3/Assign_47Assignppo_agent/ppo2_model/vf/wsave_3/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_3/RestoreV2:48*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_3/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_3/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_ff3711d50ef84dbf996d28ef4e8a2b01/part*
dtype0*
_output_shapes
: 
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_4/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
�
save_4/SaveV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
save_4/SaveV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*)
_class
loc:@save_4/ShardedFilename*
_output_shapes
: 
�
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
�
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
T0*
_output_shapes
: 
�
save_4/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_4/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_4/Assign_1Assignbeta2_powersave_4/RestoreV2:1*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_4/Assign_2Assignppo_agent/ppo2_model/pi/bsave_4/RestoreV2:2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_4/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_4/RestoreV2:3*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_4/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_4/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_4/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_4/RestoreV2:5*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_4/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_4/RestoreV2:6*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_4/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_4/RestoreV2:7*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_4/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_4/RestoreV2:8*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_4/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_4/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_4/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_4/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_4/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_4/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_4/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_4/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_4/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_4/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_4/RestoreV2:15*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_4/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_4/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_4/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_4/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_4/RestoreV2:18*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_4/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_4/RestoreV2:19*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_4/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_4/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_4/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_4/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_4/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_4/RestoreV2:22*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_4/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_4/RestoreV2:23*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_4/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_4/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_4/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_4/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_4/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_4/RestoreV2:26*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_4/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_4/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_4/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_4/RestoreV2:28*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_4/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_4/RestoreV2:29*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_4/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_4/RestoreV2:30*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_4/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_4/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_4/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_4/RestoreV2:32*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_4/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_4/RestoreV2:33*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_4/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_4/RestoreV2:34*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_4/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_4/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_4/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_4/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_4/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_4/RestoreV2:37*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_4/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_4/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_4/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_4/RestoreV2:39*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_4/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_4/RestoreV2:40*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_4/Assign_41Assignppo_agent/ppo2_model/pi/wsave_4/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_4/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_4/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_4/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_4/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_4/Assign_44Assignppo_agent/ppo2_model/vf/bsave_4/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_4/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_4/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_4/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_4/RestoreV2:46*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_4/Assign_47Assignppo_agent/ppo2_model/vf/wsave_4/RestoreV2:47*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_4/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_4/RestoreV2:48*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_4/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_4/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_40^save_4/Assign_41^save_4/Assign_42^save_4/Assign_43^save_4/Assign_44^save_4/Assign_45^save_4/Assign_46^save_4/Assign_47^save_4/Assign_48^save_4/Assign_49^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_5/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_7476be50a7d44fda9c5d1d15809c49f4/part*
_output_shapes
: 
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_5/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_5/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
�
save_5/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_5/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*)
_class
loc:@save_5/ShardedFilename*
_output_shapes
: 
�
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
�
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
�
save_5/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_5/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_5/Assign_1Assignbeta2_powersave_5/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_5/Assign_2Assignppo_agent/ppo2_model/pi/bsave_5/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_5/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_5/RestoreV2:3*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_5/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_5/RestoreV2:4*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_5/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_5/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_5/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_5/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_5/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_5/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_5/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_5/RestoreV2:8*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_5/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_5/RestoreV2:9*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_5/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_5/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_5/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_5/RestoreV2:11*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_5/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_5/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_5/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_5/RestoreV2:13*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_5/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_5/RestoreV2:14*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_5/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_5/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_5/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_5/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_5/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_5/RestoreV2:17*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_5/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_5/RestoreV2:18*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_5/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_5/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_5/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_5/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_5/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_5/RestoreV2:21*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_5/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_5/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_5/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_5/RestoreV2:23*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_5/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_5/RestoreV2:24*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_5/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_5/RestoreV2:25*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_5/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_5/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_5/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_5/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_5/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_5/RestoreV2:28*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_5/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_5/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_5/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_5/RestoreV2:30*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_5/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_5/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_5/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_5/RestoreV2:32*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_5/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_5/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_5/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_5/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_5/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_5/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_5/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_5/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_5/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_5/RestoreV2:37*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_5/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_5/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_5/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_5/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_5/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_5/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_5/Assign_41Assignppo_agent/ppo2_model/pi/wsave_5/RestoreV2:41*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_5/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_5/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_5/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_5/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_5/Assign_44Assignppo_agent/ppo2_model/vf/bsave_5/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_5/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_5/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_5/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_5/RestoreV2:46*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_5/Assign_47Assignppo_agent/ppo2_model/vf/wsave_5/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_5/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_5/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_5/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_5/RestoreV2:49*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_40^save_5/Assign_41^save_5/Assign_42^save_5/Assign_43^save_5/Assign_44^save_5/Assign_45^save_5/Assign_46^save_5/Assign_47^save_5/Assign_48^save_5/Assign_49^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_6/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_51ac2b7c19c3475b8af4a821f763890c/part*
_output_shapes
: 
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
�
save_6/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_6/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
T0*)
_class
loc:@save_6/ShardedFilename*
_output_shapes
: 
�
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(
�
save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
T0*
_output_shapes
: 
�
save_6/RestoreV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
!save_6/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_6/Assign_1Assignbeta2_powersave_6/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_6/Assign_2Assignppo_agent/ppo2_model/pi/bsave_6/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_6/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_6/RestoreV2:3*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_6/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_6/RestoreV2:4*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_6/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_6/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_6/RestoreV2:6*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_6/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_6/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_6/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_6/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_6/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_6/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_6/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_6/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_6/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_6/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_6/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_6/RestoreV2:14*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_6/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_6/RestoreV2:15*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_6/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_6/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_6/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_6/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_6/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_6/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_6/RestoreV2:20*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_6/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_6/RestoreV2:21*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_6/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_6/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_6/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_6/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_6/RestoreV2:24*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_6/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_6/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_6/RestoreV2:26*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_6/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_6/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_6/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_6/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_6/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_6/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_6/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_6/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_6/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_6/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_6/RestoreV2:33*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_6/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_6/RestoreV2:34*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_6/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_6/RestoreV2:35*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_6/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_6/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_6/RestoreV2:37*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_6/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_6/RestoreV2:38*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_6/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_6/RestoreV2:39*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_6/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_6/RestoreV2:40*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_6/Assign_41Assignppo_agent/ppo2_model/pi/wsave_6/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_6/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_6/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_6/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_6/RestoreV2:43*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_6/Assign_44Assignppo_agent/ppo2_model/vf/bsave_6/RestoreV2:44*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_6/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_6/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_6/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_6/RestoreV2:46*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_6/Assign_47Assignppo_agent/ppo2_model/vf/wsave_6/RestoreV2:47*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_6/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_6/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_6/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_6/RestoreV2:49*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_40^save_6/Assign_41^save_6/Assign_42^save_6/Assign_43^save_6/Assign_44^save_6/Assign_45^save_6/Assign_46^save_6/Assign_47^save_6/Assign_48^save_6/Assign_49^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_b034ede58bea49b2ba75dab385375133/part*
dtype0*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_7/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_7/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
�
save_7/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_7/SaveV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
T0*)
_class
loc:@save_7/ShardedFilename*
_output_shapes
: 
�
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(
�
save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
T0*
_output_shapes
: 
�
save_7/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_7/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_7/AssignAssignbeta1_powersave_7/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_7/Assign_1Assignbeta2_powersave_7/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_7/Assign_2Assignppo_agent/ppo2_model/pi/bsave_7/RestoreV2:2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_7/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_7/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_7/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_7/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_7/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_7/RestoreV2:5*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_7/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_7/RestoreV2:6*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_7/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_7/RestoreV2:7*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_7/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_7/RestoreV2:8*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_7/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_7/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_7/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_7/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_7/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_7/RestoreV2:11*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_7/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_7/RestoreV2:12*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_7/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_7/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_7/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_7/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_7/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_7/RestoreV2:15*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_7/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_7/RestoreV2:16*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_7/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_7/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_7/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_7/RestoreV2:18*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_7/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_7/RestoreV2:19*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_7/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_7/RestoreV2:20*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_7/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_7/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_7/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_7/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_7/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_7/RestoreV2:23*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_7/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_7/RestoreV2:24*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_7/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_7/RestoreV2:25*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_7/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_7/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_7/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_7/RestoreV2:27*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_7/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_7/RestoreV2:28*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_7/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_7/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_7/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_7/RestoreV2:30*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_7/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_7/RestoreV2:31*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_7/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_7/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_7/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_7/RestoreV2:33*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_7/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_7/RestoreV2:34*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_7/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_7/RestoreV2:35*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_7/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_7/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_7/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_7/RestoreV2:37*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_7/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_7/RestoreV2:38*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_7/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_7/RestoreV2:39*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_7/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_7/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_7/Assign_41Assignppo_agent/ppo2_model/pi/wsave_7/RestoreV2:41*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_7/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_7/RestoreV2:42*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_7/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_7/RestoreV2:43*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_7/Assign_44Assignppo_agent/ppo2_model/vf/bsave_7/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_7/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_7/RestoreV2:45*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_7/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_7/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_7/Assign_47Assignppo_agent/ppo2_model/vf/wsave_7/RestoreV2:47*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_7/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_7/RestoreV2:48*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_7/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_7/RestoreV2:49*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_40^save_7/Assign_41^save_7/Assign_42^save_7/Assign_43^save_7/Assign_44^save_7/Assign_45^save_7/Assign_46^save_7/Assign_47^save_7/Assign_48^save_7/Assign_49^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_8/StringJoin/inputs_1Const*<
value3B1 B+_temp_dff13c9174164a8ea9e11373cc1077ad/part*
dtype0*
_output_shapes
: 
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_8/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_8/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
�
save_8/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_8/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
T0*)
_class
loc:@save_8/ShardedFilename*
_output_shapes
: 
�
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(
�
save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
T0*
_output_shapes
: 
�
save_8/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_8/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_8/AssignAssignbeta1_powersave_8/RestoreV2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_8/Assign_1Assignbeta2_powersave_8/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_8/Assign_2Assignppo_agent/ppo2_model/pi/bsave_8/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_8/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_8/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_8/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_8/RestoreV2:4*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_8/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_8/RestoreV2:5*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_8/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_8/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_8/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_8/RestoreV2:7*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_8/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_8/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_8/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_8/RestoreV2:9*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_8/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_8/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_8/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_8/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_8/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_8/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_8/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_8/RestoreV2:13*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_8/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_8/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_8/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_8/RestoreV2:15*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_8/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_8/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_8/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_8/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_8/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_8/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_8/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_8/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_8/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_8/RestoreV2:20*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_8/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_8/RestoreV2:21*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_8/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_8/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_8/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_8/RestoreV2:23*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_8/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_8/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_8/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_8/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_8/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_8/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_8/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_8/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_8/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_8/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_8/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_8/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_8/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_8/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_8/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_8/RestoreV2:31*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_8/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_8/RestoreV2:32*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_8/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_8/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_8/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_8/RestoreV2:34*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_8/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_8/RestoreV2:35*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_8/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_8/RestoreV2:36*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_8/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_8/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_8/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_8/RestoreV2:38*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_8/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_8/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_8/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_8/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_8/Assign_41Assignppo_agent/ppo2_model/pi/wsave_8/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_8/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_8/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_8/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_8/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_8/Assign_44Assignppo_agent/ppo2_model/vf/bsave_8/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_8/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_8/RestoreV2:45*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_8/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_8/RestoreV2:46*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_8/Assign_47Assignppo_agent/ppo2_model/vf/wsave_8/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_8/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_8/RestoreV2:48*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_8/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_8/RestoreV2:49*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_40^save_8/Assign_41^save_8/Assign_42^save_8/Assign_43^save_8/Assign_44^save_8/Assign_45^save_8/Assign_46^save_8/Assign_47^save_8/Assign_48^save_8/Assign_49^save_8/Assign_5^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_9/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_3ac044c813364407a80b3ffa2196c613/part*
_output_shapes
: 
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_9/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_9/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
�
save_9/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_9/SaveV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
T0*)
_class
loc:@save_9/ShardedFilename*
_output_shapes
: 
�
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
N*
T0*

axis *
_output_shapes
:
�
save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(
�
save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
T0*
_output_shapes
: 
�
save_9/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
!save_9/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_9/AssignAssignbeta1_powersave_9/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_9/Assign_1Assignbeta2_powersave_9/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_9/Assign_2Assignppo_agent/ppo2_model/pi/bsave_9/RestoreV2:2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_9/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_9/RestoreV2:3*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_9/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_9/RestoreV2:4*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_9/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_9/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_9/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_9/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_9/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_9/RestoreV2:7*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_9/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_9/RestoreV2:8*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_9/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_9/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_9/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_9/RestoreV2:10*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_9/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_9/RestoreV2:11*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_9/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_9/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_9/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_9/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_9/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_9/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_9/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_9/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_9/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_9/RestoreV2:16*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_9/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_9/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_9/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_9/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_9/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_9/RestoreV2:19*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_9/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_9/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_9/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_9/RestoreV2:21*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_9/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_9/RestoreV2:22*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_9/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_9/RestoreV2:23*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_9/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_9/RestoreV2:24*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_9/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_9/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_9/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_9/RestoreV2:26*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_9/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_9/RestoreV2:27*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_9/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_9/RestoreV2:28*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_9/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_9/RestoreV2:29*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_9/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_9/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_9/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_9/RestoreV2:31*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_9/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_9/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_9/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_9/RestoreV2:33*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_9/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_9/RestoreV2:34*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_9/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_9/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_9/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_9/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_9/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_9/RestoreV2:37*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_9/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_9/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_9/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_9/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_9/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_9/RestoreV2:40*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_9/Assign_41Assignppo_agent/ppo2_model/pi/wsave_9/RestoreV2:41*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_9/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_9/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_9/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_9/RestoreV2:43*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_9/Assign_44Assignppo_agent/ppo2_model/vf/bsave_9/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_9/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_9/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_9/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_9/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_9/Assign_47Assignppo_agent/ppo2_model/vf/wsave_9/RestoreV2:47*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_9/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_9/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_9/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_9/RestoreV2:49*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_40^save_9/Assign_41^save_9/Assign_42^save_9/Assign_43^save_9/Assign_44^save_9/Assign_45^save_9/Assign_46^save_9/Assign_47^save_9/Assign_48^save_9/Assign_49^save_9/Assign_5^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_10/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_cea7e6e719b643108737227dba00b03e/part*
_output_shapes
: 
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_10/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_10/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
�
save_10/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_10/SaveV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2*
T0**
_class 
loc:@save_10/ShardedFilename*
_output_shapes
: 
�
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*
N*
T0*

axis *
_output_shapes
:
�
save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(
�
save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
T0*
_output_shapes
: 
�
save_10/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_10/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_10/AssignAssignbeta1_powersave_10/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_10/Assign_1Assignbeta2_powersave_10/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_10/Assign_2Assignppo_agent/ppo2_model/pi/bsave_10/RestoreV2:2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_10/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_10/RestoreV2:3*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_10/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_10/RestoreV2:4*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_10/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_10/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_10/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_10/RestoreV2:6*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_10/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_10/RestoreV2:7*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_10/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_10/RestoreV2:8*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_10/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_10/RestoreV2:9*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_10/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_10/RestoreV2:10*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_10/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_10/RestoreV2:11*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_10/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_10/RestoreV2:12*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_10/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_10/RestoreV2:13*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_10/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_10/RestoreV2:14*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_10/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_10/RestoreV2:15*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_10/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_10/RestoreV2:16*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_10/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_10/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_10/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_10/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_10/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_10/RestoreV2:19*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_10/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_10/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_10/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_10/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_10/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_10/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_10/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_10/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_10/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_10/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_10/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_10/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_10/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_10/RestoreV2:26*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_10/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_10/RestoreV2:27*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_10/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_10/RestoreV2:28*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_10/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_10/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_10/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_10/RestoreV2:30*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_10/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_10/RestoreV2:31*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_10/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_10/RestoreV2:32*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_10/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_10/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_10/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_10/RestoreV2:34*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_10/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_10/RestoreV2:35*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_10/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_10/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_10/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_10/RestoreV2:37*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_10/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_10/RestoreV2:38*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_10/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_10/RestoreV2:39*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_10/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_10/RestoreV2:40*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_10/Assign_41Assignppo_agent/ppo2_model/pi/wsave_10/RestoreV2:41*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_10/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_10/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_10/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_10/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_10/Assign_44Assignppo_agent/ppo2_model/vf/bsave_10/RestoreV2:44*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_10/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_10/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_10/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_10/RestoreV2:46*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_10/Assign_47Assignppo_agent/ppo2_model/vf/wsave_10/RestoreV2:47*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_10/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_10/RestoreV2:48*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_10/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_10/RestoreV2:49*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_40^save_10/Assign_41^save_10/Assign_42^save_10/Assign_43^save_10/Assign_44^save_10/Assign_45^save_10/Assign_46^save_10/Assign_47^save_10/Assign_48^save_10/Assign_49^save_10/Assign_5^save_10/Assign_6^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard
\
save_11/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_11/filenamePlaceholderWithDefaultsave_11/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_11/ConstPlaceholderWithDefaultsave_11/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_11/StringJoin/inputs_1Const*<
value3B1 B+_temp_b4461921e3714d8cb26fb6da4ba5fe44/part*
dtype0*
_output_shapes
: 
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_11/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_11/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
�
save_11/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_11/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2*
T0**
_class 
loc:@save_11/ShardedFilename*
_output_shapes
: 
�
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilename^save_11/control_dependency*
N*
T0*

axis *
_output_shapes
:
�
save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(
�
save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency*
T0*
_output_shapes
: 
�
save_11/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_11/RestoreV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_11/AssignAssignbeta1_powersave_11/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_11/Assign_1Assignbeta2_powersave_11/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_11/Assign_2Assignppo_agent/ppo2_model/pi/bsave_11/RestoreV2:2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_11/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_11/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_11/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_11/RestoreV2:4*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_11/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_11/RestoreV2:5*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_11/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_11/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_11/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_11/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_11/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_11/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_11/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_11/RestoreV2:9*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_11/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_11/RestoreV2:10*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_11/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_11/RestoreV2:11*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_11/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_11/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_11/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_11/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_11/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_11/RestoreV2:14*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_11/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_11/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_11/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_11/RestoreV2:16*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_11/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_11/RestoreV2:17*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_11/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_11/RestoreV2:18*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_11/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_11/RestoreV2:19*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_11/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_11/RestoreV2:20*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_11/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_11/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_11/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_11/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_11/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_11/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_11/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_11/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_11/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_11/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_11/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_11/RestoreV2:26*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_11/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_11/RestoreV2:27*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_11/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_11/RestoreV2:28*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_11/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_11/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_11/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_11/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_11/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_11/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_11/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_11/RestoreV2:32*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_11/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_11/RestoreV2:33*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_11/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_11/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_11/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_11/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_11/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_11/RestoreV2:36*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_11/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_11/RestoreV2:37*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_11/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_11/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_11/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_11/RestoreV2:39*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_11/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_11/RestoreV2:40*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_11/Assign_41Assignppo_agent/ppo2_model/pi/wsave_11/RestoreV2:41*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_11/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_11/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_11/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_11/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_11/Assign_44Assignppo_agent/ppo2_model/vf/bsave_11/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_11/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_11/RestoreV2:45*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_11/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_11/RestoreV2:46*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_11/Assign_47Assignppo_agent/ppo2_model/vf/wsave_11/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_11/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_11/RestoreV2:48*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_11/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_11/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_11/restore_shardNoOp^save_11/Assign^save_11/Assign_1^save_11/Assign_10^save_11/Assign_11^save_11/Assign_12^save_11/Assign_13^save_11/Assign_14^save_11/Assign_15^save_11/Assign_16^save_11/Assign_17^save_11/Assign_18^save_11/Assign_19^save_11/Assign_2^save_11/Assign_20^save_11/Assign_21^save_11/Assign_22^save_11/Assign_23^save_11/Assign_24^save_11/Assign_25^save_11/Assign_26^save_11/Assign_27^save_11/Assign_28^save_11/Assign_29^save_11/Assign_3^save_11/Assign_30^save_11/Assign_31^save_11/Assign_32^save_11/Assign_33^save_11/Assign_34^save_11/Assign_35^save_11/Assign_36^save_11/Assign_37^save_11/Assign_38^save_11/Assign_39^save_11/Assign_4^save_11/Assign_40^save_11/Assign_41^save_11/Assign_42^save_11/Assign_43^save_11/Assign_44^save_11/Assign_45^save_11/Assign_46^save_11/Assign_47^save_11/Assign_48^save_11/Assign_49^save_11/Assign_5^save_11/Assign_6^save_11/Assign_7^save_11/Assign_8^save_11/Assign_9
3
save_11/restore_allNoOp^save_11/restore_shard
\
save_12/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_12/filenamePlaceholderWithDefaultsave_12/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_12/ConstPlaceholderWithDefaultsave_12/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_12/StringJoin/inputs_1Const*<
value3B1 B+_temp_2f98a37d32c743059cdfc08cfcae8160/part*
dtype0*
_output_shapes
: 
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_12/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_12/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
�
save_12/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_12/SaveV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2*
T0**
_class 
loc:@save_12/ShardedFilename*
_output_shapes
: 
�
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilename^save_12/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(
�
save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency*
T0*
_output_shapes
: 
�
save_12/RestoreV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
"save_12/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_12/AssignAssignbeta1_powersave_12/RestoreV2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_12/Assign_1Assignbeta2_powersave_12/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_12/Assign_2Assignppo_agent/ppo2_model/pi/bsave_12/RestoreV2:2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_12/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_12/RestoreV2:3*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_12/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_12/RestoreV2:4*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_12/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_12/RestoreV2:5*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_12/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_12/RestoreV2:6*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_12/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_12/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_12/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_12/RestoreV2:8*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_12/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_12/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_12/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_12/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_12/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_12/RestoreV2:11*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_12/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_12/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_12/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_12/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_12/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_12/RestoreV2:14*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_12/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_12/RestoreV2:15*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_12/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_12/RestoreV2:16*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_12/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_12/RestoreV2:17*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_12/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_12/RestoreV2:18*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_12/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_12/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_12/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_12/RestoreV2:20*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_12/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_12/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_12/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_12/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_12/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_12/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_12/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_12/RestoreV2:24*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_12/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_12/RestoreV2:25*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_12/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_12/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_12/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_12/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_12/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_12/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_12/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_12/RestoreV2:29*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_12/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_12/RestoreV2:30*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_12/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_12/RestoreV2:31*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_12/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_12/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_12/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_12/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_12/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_12/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_12/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_12/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_12/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_12/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_12/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_12/RestoreV2:37*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_12/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_12/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_12/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_12/RestoreV2:39*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_12/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_12/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_12/Assign_41Assignppo_agent/ppo2_model/pi/wsave_12/RestoreV2:41*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_12/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_12/RestoreV2:42*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_12/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_12/RestoreV2:43*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_12/Assign_44Assignppo_agent/ppo2_model/vf/bsave_12/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_12/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_12/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_12/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_12/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_12/Assign_47Assignppo_agent/ppo2_model/vf/wsave_12/RestoreV2:47*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_12/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_12/RestoreV2:48*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_12/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_12/RestoreV2:49*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_16^save_12/Assign_17^save_12/Assign_18^save_12/Assign_19^save_12/Assign_2^save_12/Assign_20^save_12/Assign_21^save_12/Assign_22^save_12/Assign_23^save_12/Assign_24^save_12/Assign_25^save_12/Assign_26^save_12/Assign_27^save_12/Assign_28^save_12/Assign_29^save_12/Assign_3^save_12/Assign_30^save_12/Assign_31^save_12/Assign_32^save_12/Assign_33^save_12/Assign_34^save_12/Assign_35^save_12/Assign_36^save_12/Assign_37^save_12/Assign_38^save_12/Assign_39^save_12/Assign_4^save_12/Assign_40^save_12/Assign_41^save_12/Assign_42^save_12/Assign_43^save_12/Assign_44^save_12/Assign_45^save_12/Assign_46^save_12/Assign_47^save_12/Assign_48^save_12/Assign_49^save_12/Assign_5^save_12/Assign_6^save_12/Assign_7^save_12/Assign_8^save_12/Assign_9
3
save_12/restore_allNoOp^save_12/restore_shard
\
save_13/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_13/filenamePlaceholderWithDefaultsave_13/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_13/ConstPlaceholderWithDefaultsave_13/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_13/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_f5e8a69ca2994394a87cd09829995620/part*
_output_shapes
: 
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_13/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_13/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
�
save_13/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_13/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2*
T0**
_class 
loc:@save_13/ShardedFilename*
_output_shapes
: 
�
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilename^save_13/control_dependency*
N*
T0*

axis *
_output_shapes
:
�
save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(
�
save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency*
T0*
_output_shapes
: 
�
save_13/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_13/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_13/AssignAssignbeta1_powersave_13/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_13/Assign_1Assignbeta2_powersave_13/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_13/Assign_2Assignppo_agent/ppo2_model/pi/bsave_13/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_13/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_13/RestoreV2:3*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_13/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_13/RestoreV2:4*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_13/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_13/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_13/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_13/RestoreV2:6*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_13/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_13/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_13/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_13/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_13/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_13/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_13/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_13/RestoreV2:10*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_13/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_13/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_13/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_13/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_13/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_13/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_13/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_13/RestoreV2:14*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_13/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_13/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_13/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_13/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_13/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_13/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_13/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_13/RestoreV2:18*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_13/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_13/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_13/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_13/RestoreV2:20*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_13/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_13/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_13/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_13/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_13/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_13/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_13/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_13/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_13/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_13/RestoreV2:25*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_13/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_13/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_13/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_13/RestoreV2:27*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_13/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_13/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_13/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_13/RestoreV2:29*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_13/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_13/RestoreV2:30*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_13/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_13/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_13/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_13/RestoreV2:32*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_13/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_13/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_13/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_13/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_13/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_13/RestoreV2:35*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_13/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_13/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_13/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_13/RestoreV2:37*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_13/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_13/RestoreV2:38*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_13/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_13/RestoreV2:39*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_13/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_13/RestoreV2:40*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_13/Assign_41Assignppo_agent/ppo2_model/pi/wsave_13/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_13/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_13/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_13/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_13/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_13/Assign_44Assignppo_agent/ppo2_model/vf/bsave_13/RestoreV2:44*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_13/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_13/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_13/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_13/RestoreV2:46*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_13/Assign_47Assignppo_agent/ppo2_model/vf/wsave_13/RestoreV2:47*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_13/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_13/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_13/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_13/RestoreV2:49*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_16^save_13/Assign_17^save_13/Assign_18^save_13/Assign_19^save_13/Assign_2^save_13/Assign_20^save_13/Assign_21^save_13/Assign_22^save_13/Assign_23^save_13/Assign_24^save_13/Assign_25^save_13/Assign_26^save_13/Assign_27^save_13/Assign_28^save_13/Assign_29^save_13/Assign_3^save_13/Assign_30^save_13/Assign_31^save_13/Assign_32^save_13/Assign_33^save_13/Assign_34^save_13/Assign_35^save_13/Assign_36^save_13/Assign_37^save_13/Assign_38^save_13/Assign_39^save_13/Assign_4^save_13/Assign_40^save_13/Assign_41^save_13/Assign_42^save_13/Assign_43^save_13/Assign_44^save_13/Assign_45^save_13/Assign_46^save_13/Assign_47^save_13/Assign_48^save_13/Assign_49^save_13/Assign_5^save_13/Assign_6^save_13/Assign_7^save_13/Assign_8^save_13/Assign_9
3
save_13/restore_allNoOp^save_13/restore_shard
\
save_14/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_14/filenamePlaceholderWithDefaultsave_14/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_14/ConstPlaceholderWithDefaultsave_14/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_14/StringJoin/inputs_1Const*<
value3B1 B+_temp_a945fda74feb45e08c69d2839293bbc2/part*
dtype0*
_output_shapes
: 
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_14/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_14/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
�
save_14/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_14/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2*
T0**
_class 
loc:@save_14/ShardedFilename*
_output_shapes
: 
�
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilename^save_14/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(
�
save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency*
T0*
_output_shapes
: 
�
save_14/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_14/RestoreV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_14/AssignAssignbeta1_powersave_14/RestoreV2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_14/Assign_1Assignbeta2_powersave_14/RestoreV2:1*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_14/Assign_2Assignppo_agent/ppo2_model/pi/bsave_14/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_14/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_14/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_14/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_14/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_14/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_14/RestoreV2:5*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_14/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_14/RestoreV2:6*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_14/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_14/RestoreV2:7*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_14/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_14/RestoreV2:8*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_14/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_14/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_14/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_14/RestoreV2:10*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_14/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_14/RestoreV2:11*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_14/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_14/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_14/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_14/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_14/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_14/RestoreV2:14*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_14/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_14/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_14/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_14/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_14/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_14/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_14/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_14/RestoreV2:18*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_14/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_14/RestoreV2:19*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_14/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_14/RestoreV2:20*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_14/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_14/RestoreV2:21*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_14/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_14/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_14/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_14/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_14/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_14/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_14/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_14/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_14/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_14/RestoreV2:26*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_14/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_14/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_14/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_14/RestoreV2:28*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_14/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_14/RestoreV2:29*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_14/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_14/RestoreV2:30*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_14/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_14/RestoreV2:31*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_14/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_14/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_14/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_14/RestoreV2:33*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_14/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_14/RestoreV2:34*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_14/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_14/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_14/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_14/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_14/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_14/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_14/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_14/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_14/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_14/RestoreV2:39*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_14/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_14/RestoreV2:40*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_14/Assign_41Assignppo_agent/ppo2_model/pi/wsave_14/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_14/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_14/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_14/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_14/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_14/Assign_44Assignppo_agent/ppo2_model/vf/bsave_14/RestoreV2:44*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_14/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_14/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_14/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_14/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_14/Assign_47Assignppo_agent/ppo2_model/vf/wsave_14/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_14/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_14/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_14/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_14/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_16^save_14/Assign_17^save_14/Assign_18^save_14/Assign_19^save_14/Assign_2^save_14/Assign_20^save_14/Assign_21^save_14/Assign_22^save_14/Assign_23^save_14/Assign_24^save_14/Assign_25^save_14/Assign_26^save_14/Assign_27^save_14/Assign_28^save_14/Assign_29^save_14/Assign_3^save_14/Assign_30^save_14/Assign_31^save_14/Assign_32^save_14/Assign_33^save_14/Assign_34^save_14/Assign_35^save_14/Assign_36^save_14/Assign_37^save_14/Assign_38^save_14/Assign_39^save_14/Assign_4^save_14/Assign_40^save_14/Assign_41^save_14/Assign_42^save_14/Assign_43^save_14/Assign_44^save_14/Assign_45^save_14/Assign_46^save_14/Assign_47^save_14/Assign_48^save_14/Assign_49^save_14/Assign_5^save_14/Assign_6^save_14/Assign_7^save_14/Assign_8^save_14/Assign_9
3
save_14/restore_allNoOp^save_14/restore_shard
\
save_15/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_15/filenamePlaceholderWithDefaultsave_15/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_15/ConstPlaceholderWithDefaultsave_15/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_15/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_026b0200bcf049d59cf1f1bd14164188/part*
_output_shapes
: 
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_15/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_15/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
�
save_15/SaveV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
save_15/SaveV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2*
T0**
_class 
loc:@save_15/ShardedFilename*
_output_shapes
: 
�
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilename^save_15/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(
�
save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency*
T0*
_output_shapes
: 
�
save_15/RestoreV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
"save_15/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_15/AssignAssignbeta1_powersave_15/RestoreV2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_15/Assign_1Assignbeta2_powersave_15/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_15/Assign_2Assignppo_agent/ppo2_model/pi/bsave_15/RestoreV2:2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_15/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_15/RestoreV2:3*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_15/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_15/RestoreV2:4*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_15/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_15/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_15/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_15/RestoreV2:6*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_15/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_15/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_15/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_15/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_15/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_15/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_15/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_15/RestoreV2:10*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_15/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_15/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_15/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_15/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_15/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_15/RestoreV2:13*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_15/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_15/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_15/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_15/RestoreV2:15*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_15/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_15/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_15/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_15/RestoreV2:17*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_15/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_15/RestoreV2:18*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_15/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_15/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_15/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_15/RestoreV2:20*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_15/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_15/RestoreV2:21*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_15/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_15/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_15/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_15/RestoreV2:23*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_15/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_15/RestoreV2:24*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_15/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_15/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_15/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_15/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_15/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_15/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_15/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_15/RestoreV2:28*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_15/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_15/RestoreV2:29*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_15/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_15/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_15/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_15/RestoreV2:31*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_15/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_15/RestoreV2:32*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_15/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_15/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_15/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_15/RestoreV2:34*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_15/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_15/RestoreV2:35*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_15/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_15/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_15/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_15/RestoreV2:37*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_15/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_15/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_15/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_15/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_15/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_15/RestoreV2:40*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_15/Assign_41Assignppo_agent/ppo2_model/pi/wsave_15/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_15/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_15/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_15/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_15/RestoreV2:43*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_15/Assign_44Assignppo_agent/ppo2_model/vf/bsave_15/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_15/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_15/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_15/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_15/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_15/Assign_47Assignppo_agent/ppo2_model/vf/wsave_15/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_15/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_15/RestoreV2:48*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_15/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_15/RestoreV2:49*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_15/restore_shardNoOp^save_15/Assign^save_15/Assign_1^save_15/Assign_10^save_15/Assign_11^save_15/Assign_12^save_15/Assign_13^save_15/Assign_14^save_15/Assign_15^save_15/Assign_16^save_15/Assign_17^save_15/Assign_18^save_15/Assign_19^save_15/Assign_2^save_15/Assign_20^save_15/Assign_21^save_15/Assign_22^save_15/Assign_23^save_15/Assign_24^save_15/Assign_25^save_15/Assign_26^save_15/Assign_27^save_15/Assign_28^save_15/Assign_29^save_15/Assign_3^save_15/Assign_30^save_15/Assign_31^save_15/Assign_32^save_15/Assign_33^save_15/Assign_34^save_15/Assign_35^save_15/Assign_36^save_15/Assign_37^save_15/Assign_38^save_15/Assign_39^save_15/Assign_4^save_15/Assign_40^save_15/Assign_41^save_15/Assign_42^save_15/Assign_43^save_15/Assign_44^save_15/Assign_45^save_15/Assign_46^save_15/Assign_47^save_15/Assign_48^save_15/Assign_49^save_15/Assign_5^save_15/Assign_6^save_15/Assign_7^save_15/Assign_8^save_15/Assign_9
3
save_15/restore_allNoOp^save_15/restore_shard
\
save_16/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_16/filenamePlaceholderWithDefaultsave_16/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_16/ConstPlaceholderWithDefaultsave_16/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_16/StringJoin/inputs_1Const*<
value3B1 B+_temp_e9c222351db44f359959e9b0b97e58ea/part*
dtype0*
_output_shapes
: 
~
save_16/StringJoin
StringJoinsave_16/Constsave_16/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_16/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_16/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_16/ShardedFilenameShardedFilenamesave_16/StringJoinsave_16/ShardedFilename/shardsave_16/num_shards*
_output_shapes
: 
�
save_16/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_16/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_16/SaveV2SaveV2save_16/ShardedFilenamesave_16/SaveV2/tensor_namessave_16/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_16/control_dependencyIdentitysave_16/ShardedFilename^save_16/SaveV2*
T0**
_class 
loc:@save_16/ShardedFilename*
_output_shapes
: 
�
.save_16/MergeV2Checkpoints/checkpoint_prefixesPacksave_16/ShardedFilename^save_16/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_16/MergeV2CheckpointsMergeV2Checkpoints.save_16/MergeV2Checkpoints/checkpoint_prefixessave_16/Const*
delete_old_dirs(
�
save_16/IdentityIdentitysave_16/Const^save_16/MergeV2Checkpoints^save_16/control_dependency*
T0*
_output_shapes
: 
�
save_16/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_16/RestoreV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_16/RestoreV2	RestoreV2save_16/Constsave_16/RestoreV2/tensor_names"save_16/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_16/AssignAssignbeta1_powersave_16/RestoreV2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_16/Assign_1Assignbeta2_powersave_16/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_16/Assign_2Assignppo_agent/ppo2_model/pi/bsave_16/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_16/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_16/RestoreV2:3*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_16/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_16/RestoreV2:4*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_16/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_16/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_16/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_16/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_16/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_16/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_16/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_16/RestoreV2:8*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_16/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_16/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_16/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_16/RestoreV2:10*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_16/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_16/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_16/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_16/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_16/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_16/RestoreV2:13*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_16/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_16/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_16/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_16/RestoreV2:15*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_16/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_16/RestoreV2:16*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_16/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_16/RestoreV2:17*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_16/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_16/RestoreV2:18*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_16/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_16/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_16/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_16/RestoreV2:20*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_16/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_16/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_16/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_16/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_16/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_16/RestoreV2:23*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_16/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_16/RestoreV2:24*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_16/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_16/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_16/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_16/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_16/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_16/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_16/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_16/RestoreV2:28*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_16/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_16/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_16/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_16/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_16/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_16/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_16/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_16/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_16/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_16/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_16/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_16/RestoreV2:34*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_16/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_16/RestoreV2:35*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_16/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_16/RestoreV2:36*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_16/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_16/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_16/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_16/RestoreV2:38*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_16/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_16/RestoreV2:39*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_16/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_16/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_16/Assign_41Assignppo_agent/ppo2_model/pi/wsave_16/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_16/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_16/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_16/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_16/RestoreV2:43*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_16/Assign_44Assignppo_agent/ppo2_model/vf/bsave_16/RestoreV2:44*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_16/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_16/RestoreV2:45*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_16/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_16/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_16/Assign_47Assignppo_agent/ppo2_model/vf/wsave_16/RestoreV2:47*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_16/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_16/RestoreV2:48*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_16/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_16/RestoreV2:49*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_16/restore_shardNoOp^save_16/Assign^save_16/Assign_1^save_16/Assign_10^save_16/Assign_11^save_16/Assign_12^save_16/Assign_13^save_16/Assign_14^save_16/Assign_15^save_16/Assign_16^save_16/Assign_17^save_16/Assign_18^save_16/Assign_19^save_16/Assign_2^save_16/Assign_20^save_16/Assign_21^save_16/Assign_22^save_16/Assign_23^save_16/Assign_24^save_16/Assign_25^save_16/Assign_26^save_16/Assign_27^save_16/Assign_28^save_16/Assign_29^save_16/Assign_3^save_16/Assign_30^save_16/Assign_31^save_16/Assign_32^save_16/Assign_33^save_16/Assign_34^save_16/Assign_35^save_16/Assign_36^save_16/Assign_37^save_16/Assign_38^save_16/Assign_39^save_16/Assign_4^save_16/Assign_40^save_16/Assign_41^save_16/Assign_42^save_16/Assign_43^save_16/Assign_44^save_16/Assign_45^save_16/Assign_46^save_16/Assign_47^save_16/Assign_48^save_16/Assign_49^save_16/Assign_5^save_16/Assign_6^save_16/Assign_7^save_16/Assign_8^save_16/Assign_9
3
save_16/restore_allNoOp^save_16/restore_shard
\
save_17/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_17/filenamePlaceholderWithDefaultsave_17/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_17/ConstPlaceholderWithDefaultsave_17/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_17/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_3cd726b6c8224255b0bcd938e38377c2/part*
_output_shapes
: 
~
save_17/StringJoin
StringJoinsave_17/Constsave_17/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_17/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_17/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_17/ShardedFilenameShardedFilenamesave_17/StringJoinsave_17/ShardedFilename/shardsave_17/num_shards*
_output_shapes
: 
�
save_17/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_17/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_17/SaveV2SaveV2save_17/ShardedFilenamesave_17/SaveV2/tensor_namessave_17/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_17/control_dependencyIdentitysave_17/ShardedFilename^save_17/SaveV2*
T0**
_class 
loc:@save_17/ShardedFilename*
_output_shapes
: 
�
.save_17/MergeV2Checkpoints/checkpoint_prefixesPacksave_17/ShardedFilename^save_17/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_17/MergeV2CheckpointsMergeV2Checkpoints.save_17/MergeV2Checkpoints/checkpoint_prefixessave_17/Const*
delete_old_dirs(
�
save_17/IdentityIdentitysave_17/Const^save_17/MergeV2Checkpoints^save_17/control_dependency*
T0*
_output_shapes
: 
�
save_17/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_17/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_17/RestoreV2	RestoreV2save_17/Constsave_17/RestoreV2/tensor_names"save_17/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_17/AssignAssignbeta1_powersave_17/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_17/Assign_1Assignbeta2_powersave_17/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_17/Assign_2Assignppo_agent/ppo2_model/pi/bsave_17/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_17/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_17/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_17/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_17/RestoreV2:4*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_17/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_17/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_17/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_17/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_17/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_17/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_17/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_17/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_17/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_17/RestoreV2:9*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_17/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_17/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_17/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_17/RestoreV2:11*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_17/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_17/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_17/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_17/RestoreV2:13*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_17/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_17/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_17/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_17/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_17/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_17/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_17/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_17/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_17/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_17/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_17/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_17/RestoreV2:19*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_17/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_17/RestoreV2:20*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_17/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_17/RestoreV2:21*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_17/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_17/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_17/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_17/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_17/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_17/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_17/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_17/RestoreV2:25*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_17/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_17/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_17/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_17/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_17/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_17/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_17/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_17/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_17/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_17/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_17/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_17/RestoreV2:31*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_17/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_17/RestoreV2:32*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_17/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_17/RestoreV2:33*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_17/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_17/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_17/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_17/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_17/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_17/RestoreV2:36*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_17/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_17/RestoreV2:37*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_17/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_17/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_17/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_17/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_17/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_17/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_17/Assign_41Assignppo_agent/ppo2_model/pi/wsave_17/RestoreV2:41*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_17/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_17/RestoreV2:42*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_17/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_17/RestoreV2:43*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_17/Assign_44Assignppo_agent/ppo2_model/vf/bsave_17/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_17/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_17/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_17/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_17/RestoreV2:46*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_17/Assign_47Assignppo_agent/ppo2_model/vf/wsave_17/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_17/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_17/RestoreV2:48*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_17/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_17/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_17/restore_shardNoOp^save_17/Assign^save_17/Assign_1^save_17/Assign_10^save_17/Assign_11^save_17/Assign_12^save_17/Assign_13^save_17/Assign_14^save_17/Assign_15^save_17/Assign_16^save_17/Assign_17^save_17/Assign_18^save_17/Assign_19^save_17/Assign_2^save_17/Assign_20^save_17/Assign_21^save_17/Assign_22^save_17/Assign_23^save_17/Assign_24^save_17/Assign_25^save_17/Assign_26^save_17/Assign_27^save_17/Assign_28^save_17/Assign_29^save_17/Assign_3^save_17/Assign_30^save_17/Assign_31^save_17/Assign_32^save_17/Assign_33^save_17/Assign_34^save_17/Assign_35^save_17/Assign_36^save_17/Assign_37^save_17/Assign_38^save_17/Assign_39^save_17/Assign_4^save_17/Assign_40^save_17/Assign_41^save_17/Assign_42^save_17/Assign_43^save_17/Assign_44^save_17/Assign_45^save_17/Assign_46^save_17/Assign_47^save_17/Assign_48^save_17/Assign_49^save_17/Assign_5^save_17/Assign_6^save_17/Assign_7^save_17/Assign_8^save_17/Assign_9
3
save_17/restore_allNoOp^save_17/restore_shard
\
save_18/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_18/filenamePlaceholderWithDefaultsave_18/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_18/ConstPlaceholderWithDefaultsave_18/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_18/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_b4ddc28f365b409182c652464f767d8b/part*
_output_shapes
: 
~
save_18/StringJoin
StringJoinsave_18/Constsave_18/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_18/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_18/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_18/ShardedFilenameShardedFilenamesave_18/StringJoinsave_18/ShardedFilename/shardsave_18/num_shards*
_output_shapes
: 
�
save_18/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_18/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_18/SaveV2SaveV2save_18/ShardedFilenamesave_18/SaveV2/tensor_namessave_18/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_18/control_dependencyIdentitysave_18/ShardedFilename^save_18/SaveV2*
T0**
_class 
loc:@save_18/ShardedFilename*
_output_shapes
: 
�
.save_18/MergeV2Checkpoints/checkpoint_prefixesPacksave_18/ShardedFilename^save_18/control_dependency*
N*
T0*

axis *
_output_shapes
:
�
save_18/MergeV2CheckpointsMergeV2Checkpoints.save_18/MergeV2Checkpoints/checkpoint_prefixessave_18/Const*
delete_old_dirs(
�
save_18/IdentityIdentitysave_18/Const^save_18/MergeV2Checkpoints^save_18/control_dependency*
T0*
_output_shapes
: 
�
save_18/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_18/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_18/RestoreV2	RestoreV2save_18/Constsave_18/RestoreV2/tensor_names"save_18/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_18/AssignAssignbeta1_powersave_18/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_18/Assign_1Assignbeta2_powersave_18/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_18/Assign_2Assignppo_agent/ppo2_model/pi/bsave_18/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_18/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_18/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_18/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_18/RestoreV2:4*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_18/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_18/RestoreV2:5*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_18/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_18/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_18/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_18/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_18/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_18/RestoreV2:8*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_18/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_18/RestoreV2:9*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_18/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_18/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_18/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_18/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_18/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_18/RestoreV2:12*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_18/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_18/RestoreV2:13*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_18/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_18/RestoreV2:14*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_18/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_18/RestoreV2:15*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_18/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_18/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_18/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_18/RestoreV2:17*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_18/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_18/RestoreV2:18*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_18/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_18/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_18/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_18/RestoreV2:20*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_18/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_18/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_18/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_18/RestoreV2:22*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_18/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_18/RestoreV2:23*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_18/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_18/RestoreV2:24*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_18/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_18/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_18/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_18/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_18/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_18/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_18/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_18/RestoreV2:28*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_18/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_18/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_18/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_18/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_18/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_18/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_18/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_18/RestoreV2:32*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_18/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_18/RestoreV2:33*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_18/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_18/RestoreV2:34*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_18/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_18/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_18/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_18/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_18/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_18/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_18/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_18/RestoreV2:38*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_18/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_18/RestoreV2:39*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_18/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_18/RestoreV2:40*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_18/Assign_41Assignppo_agent/ppo2_model/pi/wsave_18/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_18/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_18/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_18/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_18/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_18/Assign_44Assignppo_agent/ppo2_model/vf/bsave_18/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_18/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_18/RestoreV2:45*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_18/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_18/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_18/Assign_47Assignppo_agent/ppo2_model/vf/wsave_18/RestoreV2:47*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_18/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_18/RestoreV2:48*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_18/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_18/RestoreV2:49*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_18/restore_shardNoOp^save_18/Assign^save_18/Assign_1^save_18/Assign_10^save_18/Assign_11^save_18/Assign_12^save_18/Assign_13^save_18/Assign_14^save_18/Assign_15^save_18/Assign_16^save_18/Assign_17^save_18/Assign_18^save_18/Assign_19^save_18/Assign_2^save_18/Assign_20^save_18/Assign_21^save_18/Assign_22^save_18/Assign_23^save_18/Assign_24^save_18/Assign_25^save_18/Assign_26^save_18/Assign_27^save_18/Assign_28^save_18/Assign_29^save_18/Assign_3^save_18/Assign_30^save_18/Assign_31^save_18/Assign_32^save_18/Assign_33^save_18/Assign_34^save_18/Assign_35^save_18/Assign_36^save_18/Assign_37^save_18/Assign_38^save_18/Assign_39^save_18/Assign_4^save_18/Assign_40^save_18/Assign_41^save_18/Assign_42^save_18/Assign_43^save_18/Assign_44^save_18/Assign_45^save_18/Assign_46^save_18/Assign_47^save_18/Assign_48^save_18/Assign_49^save_18/Assign_5^save_18/Assign_6^save_18/Assign_7^save_18/Assign_8^save_18/Assign_9
3
save_18/restore_allNoOp^save_18/restore_shard
\
save_19/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_19/filenamePlaceholderWithDefaultsave_19/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_19/ConstPlaceholderWithDefaultsave_19/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_19/StringJoin/inputs_1Const*<
value3B1 B+_temp_5db422cb479347be8aa01bb7b230c3c8/part*
dtype0*
_output_shapes
: 
~
save_19/StringJoin
StringJoinsave_19/Constsave_19/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_19/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_19/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_19/ShardedFilenameShardedFilenamesave_19/StringJoinsave_19/ShardedFilename/shardsave_19/num_shards*
_output_shapes
: 
�
save_19/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_19/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_19/SaveV2SaveV2save_19/ShardedFilenamesave_19/SaveV2/tensor_namessave_19/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_19/control_dependencyIdentitysave_19/ShardedFilename^save_19/SaveV2*
T0**
_class 
loc:@save_19/ShardedFilename*
_output_shapes
: 
�
.save_19/MergeV2Checkpoints/checkpoint_prefixesPacksave_19/ShardedFilename^save_19/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_19/MergeV2CheckpointsMergeV2Checkpoints.save_19/MergeV2Checkpoints/checkpoint_prefixessave_19/Const*
delete_old_dirs(
�
save_19/IdentityIdentitysave_19/Const^save_19/MergeV2Checkpoints^save_19/control_dependency*
T0*
_output_shapes
: 
�
save_19/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_19/RestoreV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_19/RestoreV2	RestoreV2save_19/Constsave_19/RestoreV2/tensor_names"save_19/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_19/AssignAssignbeta1_powersave_19/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_19/Assign_1Assignbeta2_powersave_19/RestoreV2:1*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_19/Assign_2Assignppo_agent/ppo2_model/pi/bsave_19/RestoreV2:2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_19/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_19/RestoreV2:3*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_19/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_19/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_19/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_19/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_19/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_19/RestoreV2:6*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_19/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_19/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_19/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_19/RestoreV2:8*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_19/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_19/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_19/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_19/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_19/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_19/RestoreV2:11*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_19/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_19/RestoreV2:12*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_19/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_19/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_19/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_19/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_19/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_19/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_19/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_19/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_19/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_19/RestoreV2:17*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_19/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_19/RestoreV2:18*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_19/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_19/RestoreV2:19*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_19/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_19/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_19/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_19/RestoreV2:21*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_19/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_19/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_19/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_19/RestoreV2:23*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_19/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_19/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_19/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_19/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_19/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_19/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_19/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_19/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_19/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_19/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_19/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_19/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_19/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_19/RestoreV2:30*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_19/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_19/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_19/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_19/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_19/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_19/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_19/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_19/RestoreV2:34*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_19/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_19/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_19/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_19/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_19/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_19/RestoreV2:37*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_19/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_19/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_19/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_19/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_19/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_19/RestoreV2:40*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_19/Assign_41Assignppo_agent/ppo2_model/pi/wsave_19/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_19/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_19/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_19/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_19/RestoreV2:43*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_19/Assign_44Assignppo_agent/ppo2_model/vf/bsave_19/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_19/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_19/RestoreV2:45*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_19/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_19/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_19/Assign_47Assignppo_agent/ppo2_model/vf/wsave_19/RestoreV2:47*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_19/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_19/RestoreV2:48*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_19/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_19/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_19/restore_shardNoOp^save_19/Assign^save_19/Assign_1^save_19/Assign_10^save_19/Assign_11^save_19/Assign_12^save_19/Assign_13^save_19/Assign_14^save_19/Assign_15^save_19/Assign_16^save_19/Assign_17^save_19/Assign_18^save_19/Assign_19^save_19/Assign_2^save_19/Assign_20^save_19/Assign_21^save_19/Assign_22^save_19/Assign_23^save_19/Assign_24^save_19/Assign_25^save_19/Assign_26^save_19/Assign_27^save_19/Assign_28^save_19/Assign_29^save_19/Assign_3^save_19/Assign_30^save_19/Assign_31^save_19/Assign_32^save_19/Assign_33^save_19/Assign_34^save_19/Assign_35^save_19/Assign_36^save_19/Assign_37^save_19/Assign_38^save_19/Assign_39^save_19/Assign_4^save_19/Assign_40^save_19/Assign_41^save_19/Assign_42^save_19/Assign_43^save_19/Assign_44^save_19/Assign_45^save_19/Assign_46^save_19/Assign_47^save_19/Assign_48^save_19/Assign_49^save_19/Assign_5^save_19/Assign_6^save_19/Assign_7^save_19/Assign_8^save_19/Assign_9
3
save_19/restore_allNoOp^save_19/restore_shard
\
save_20/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_20/filenamePlaceholderWithDefaultsave_20/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_20/ConstPlaceholderWithDefaultsave_20/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_20/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_8dfa3da2fae0483f9c0479cffe1d2170/part*
_output_shapes
: 
~
save_20/StringJoin
StringJoinsave_20/Constsave_20/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_20/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_20/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_20/ShardedFilenameShardedFilenamesave_20/StringJoinsave_20/ShardedFilename/shardsave_20/num_shards*
_output_shapes
: 
�
save_20/SaveV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
save_20/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_20/SaveV2SaveV2save_20/ShardedFilenamesave_20/SaveV2/tensor_namessave_20/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_20/control_dependencyIdentitysave_20/ShardedFilename^save_20/SaveV2*
T0**
_class 
loc:@save_20/ShardedFilename*
_output_shapes
: 
�
.save_20/MergeV2Checkpoints/checkpoint_prefixesPacksave_20/ShardedFilename^save_20/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_20/MergeV2CheckpointsMergeV2Checkpoints.save_20/MergeV2Checkpoints/checkpoint_prefixessave_20/Const*
delete_old_dirs(
�
save_20/IdentityIdentitysave_20/Const^save_20/MergeV2Checkpoints^save_20/control_dependency*
T0*
_output_shapes
: 
�
save_20/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_20/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_20/RestoreV2	RestoreV2save_20/Constsave_20/RestoreV2/tensor_names"save_20/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_20/AssignAssignbeta1_powersave_20/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_20/Assign_1Assignbeta2_powersave_20/RestoreV2:1*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_20/Assign_2Assignppo_agent/ppo2_model/pi/bsave_20/RestoreV2:2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_20/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_20/RestoreV2:3*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_20/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_20/RestoreV2:4*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_20/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_20/RestoreV2:5*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_20/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_20/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_20/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_20/RestoreV2:7*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_20/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_20/RestoreV2:8*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_20/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_20/RestoreV2:9*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_20/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_20/RestoreV2:10*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_20/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_20/RestoreV2:11*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_20/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_20/RestoreV2:12*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_20/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_20/RestoreV2:13*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_20/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_20/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_20/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_20/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_20/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_20/RestoreV2:16*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_20/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_20/RestoreV2:17*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_20/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_20/RestoreV2:18*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_20/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_20/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_20/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_20/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_20/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_20/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_20/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_20/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_20/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_20/RestoreV2:23*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_20/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_20/RestoreV2:24*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_20/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_20/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_20/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_20/RestoreV2:26*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_20/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_20/RestoreV2:27*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_20/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_20/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_20/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_20/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_20/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_20/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_20/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_20/RestoreV2:31*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_20/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_20/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_20/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_20/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_20/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_20/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_20/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_20/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_20/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_20/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_20/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_20/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_20/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_20/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_20/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_20/RestoreV2:39*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_20/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_20/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_20/Assign_41Assignppo_agent/ppo2_model/pi/wsave_20/RestoreV2:41*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_20/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_20/RestoreV2:42*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_20/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_20/RestoreV2:43*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_20/Assign_44Assignppo_agent/ppo2_model/vf/bsave_20/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_20/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_20/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_20/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_20/RestoreV2:46*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_20/Assign_47Assignppo_agent/ppo2_model/vf/wsave_20/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_20/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_20/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_20/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_20/RestoreV2:49*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_20/restore_shardNoOp^save_20/Assign^save_20/Assign_1^save_20/Assign_10^save_20/Assign_11^save_20/Assign_12^save_20/Assign_13^save_20/Assign_14^save_20/Assign_15^save_20/Assign_16^save_20/Assign_17^save_20/Assign_18^save_20/Assign_19^save_20/Assign_2^save_20/Assign_20^save_20/Assign_21^save_20/Assign_22^save_20/Assign_23^save_20/Assign_24^save_20/Assign_25^save_20/Assign_26^save_20/Assign_27^save_20/Assign_28^save_20/Assign_29^save_20/Assign_3^save_20/Assign_30^save_20/Assign_31^save_20/Assign_32^save_20/Assign_33^save_20/Assign_34^save_20/Assign_35^save_20/Assign_36^save_20/Assign_37^save_20/Assign_38^save_20/Assign_39^save_20/Assign_4^save_20/Assign_40^save_20/Assign_41^save_20/Assign_42^save_20/Assign_43^save_20/Assign_44^save_20/Assign_45^save_20/Assign_46^save_20/Assign_47^save_20/Assign_48^save_20/Assign_49^save_20/Assign_5^save_20/Assign_6^save_20/Assign_7^save_20/Assign_8^save_20/Assign_9
3
save_20/restore_allNoOp^save_20/restore_shard
\
save_21/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_21/filenamePlaceholderWithDefaultsave_21/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_21/ConstPlaceholderWithDefaultsave_21/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_21/StringJoin/inputs_1Const*<
value3B1 B+_temp_649cfe19ddeb464bad48ec3d191510d2/part*
dtype0*
_output_shapes
: 
~
save_21/StringJoin
StringJoinsave_21/Constsave_21/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_21/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_21/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_21/ShardedFilenameShardedFilenamesave_21/StringJoinsave_21/ShardedFilename/shardsave_21/num_shards*
_output_shapes
: 
�
save_21/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_21/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_21/SaveV2SaveV2save_21/ShardedFilenamesave_21/SaveV2/tensor_namessave_21/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_21/control_dependencyIdentitysave_21/ShardedFilename^save_21/SaveV2*
T0**
_class 
loc:@save_21/ShardedFilename*
_output_shapes
: 
�
.save_21/MergeV2Checkpoints/checkpoint_prefixesPacksave_21/ShardedFilename^save_21/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_21/MergeV2CheckpointsMergeV2Checkpoints.save_21/MergeV2Checkpoints/checkpoint_prefixessave_21/Const*
delete_old_dirs(
�
save_21/IdentityIdentitysave_21/Const^save_21/MergeV2Checkpoints^save_21/control_dependency*
T0*
_output_shapes
: 
�
save_21/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_21/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_21/RestoreV2	RestoreV2save_21/Constsave_21/RestoreV2/tensor_names"save_21/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_21/AssignAssignbeta1_powersave_21/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_21/Assign_1Assignbeta2_powersave_21/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_21/Assign_2Assignppo_agent/ppo2_model/pi/bsave_21/RestoreV2:2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_21/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_21/RestoreV2:3*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_21/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_21/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_21/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_21/RestoreV2:5*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_21/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_21/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_21/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_21/RestoreV2:7*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_21/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_21/RestoreV2:8*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_21/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_21/RestoreV2:9*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_21/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_21/RestoreV2:10*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_21/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_21/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_21/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_21/RestoreV2:12*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_21/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_21/RestoreV2:13*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_21/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_21/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_21/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_21/RestoreV2:15*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_21/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_21/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_21/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_21/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_21/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_21/RestoreV2:18*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_21/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_21/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_21/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_21/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_21/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_21/RestoreV2:21*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_21/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_21/RestoreV2:22*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_21/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_21/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_21/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_21/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_21/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_21/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_21/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_21/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_21/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_21/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_21/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_21/RestoreV2:28*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_21/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_21/RestoreV2:29*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_21/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_21/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_21/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_21/RestoreV2:31*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_21/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_21/RestoreV2:32*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_21/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_21/RestoreV2:33*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_21/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_21/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_21/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_21/RestoreV2:35*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_21/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_21/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_21/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_21/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_21/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_21/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_21/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_21/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_21/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_21/RestoreV2:40*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_21/Assign_41Assignppo_agent/ppo2_model/pi/wsave_21/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_21/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_21/RestoreV2:42*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_21/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_21/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_21/Assign_44Assignppo_agent/ppo2_model/vf/bsave_21/RestoreV2:44*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_21/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_21/RestoreV2:45*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_21/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_21/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_21/Assign_47Assignppo_agent/ppo2_model/vf/wsave_21/RestoreV2:47*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_21/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_21/RestoreV2:48*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_21/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_21/RestoreV2:49*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_21/restore_shardNoOp^save_21/Assign^save_21/Assign_1^save_21/Assign_10^save_21/Assign_11^save_21/Assign_12^save_21/Assign_13^save_21/Assign_14^save_21/Assign_15^save_21/Assign_16^save_21/Assign_17^save_21/Assign_18^save_21/Assign_19^save_21/Assign_2^save_21/Assign_20^save_21/Assign_21^save_21/Assign_22^save_21/Assign_23^save_21/Assign_24^save_21/Assign_25^save_21/Assign_26^save_21/Assign_27^save_21/Assign_28^save_21/Assign_29^save_21/Assign_3^save_21/Assign_30^save_21/Assign_31^save_21/Assign_32^save_21/Assign_33^save_21/Assign_34^save_21/Assign_35^save_21/Assign_36^save_21/Assign_37^save_21/Assign_38^save_21/Assign_39^save_21/Assign_4^save_21/Assign_40^save_21/Assign_41^save_21/Assign_42^save_21/Assign_43^save_21/Assign_44^save_21/Assign_45^save_21/Assign_46^save_21/Assign_47^save_21/Assign_48^save_21/Assign_49^save_21/Assign_5^save_21/Assign_6^save_21/Assign_7^save_21/Assign_8^save_21/Assign_9
3
save_21/restore_allNoOp^save_21/restore_shard
\
save_22/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_22/filenamePlaceholderWithDefaultsave_22/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_22/ConstPlaceholderWithDefaultsave_22/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_22/StringJoin/inputs_1Const*<
value3B1 B+_temp_8fa29bb2b1ff4881a2c550f535bd547f/part*
dtype0*
_output_shapes
: 
~
save_22/StringJoin
StringJoinsave_22/Constsave_22/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_22/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_22/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_22/ShardedFilenameShardedFilenamesave_22/StringJoinsave_22/ShardedFilename/shardsave_22/num_shards*
_output_shapes
: 
�
save_22/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_22/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_22/SaveV2SaveV2save_22/ShardedFilenamesave_22/SaveV2/tensor_namessave_22/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_22/control_dependencyIdentitysave_22/ShardedFilename^save_22/SaveV2*
T0**
_class 
loc:@save_22/ShardedFilename*
_output_shapes
: 
�
.save_22/MergeV2Checkpoints/checkpoint_prefixesPacksave_22/ShardedFilename^save_22/control_dependency*
N*
T0*

axis *
_output_shapes
:
�
save_22/MergeV2CheckpointsMergeV2Checkpoints.save_22/MergeV2Checkpoints/checkpoint_prefixessave_22/Const*
delete_old_dirs(
�
save_22/IdentityIdentitysave_22/Const^save_22/MergeV2Checkpoints^save_22/control_dependency*
T0*
_output_shapes
: 
�
save_22/RestoreV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
"save_22/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_22/RestoreV2	RestoreV2save_22/Constsave_22/RestoreV2/tensor_names"save_22/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_22/AssignAssignbeta1_powersave_22/RestoreV2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_22/Assign_1Assignbeta2_powersave_22/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_22/Assign_2Assignppo_agent/ppo2_model/pi/bsave_22/RestoreV2:2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_22/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_22/RestoreV2:3*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_22/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_22/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_22/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_22/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_22/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_22/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_22/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_22/RestoreV2:7*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_22/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_22/RestoreV2:8*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_22/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_22/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_22/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_22/RestoreV2:10*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_22/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_22/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_22/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_22/RestoreV2:12*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_22/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_22/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_22/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_22/RestoreV2:14*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_22/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_22/RestoreV2:15*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_22/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_22/RestoreV2:16*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_22/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_22/RestoreV2:17*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_22/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_22/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_22/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_22/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_22/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_22/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_22/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_22/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_22/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_22/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_22/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_22/RestoreV2:23*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_22/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_22/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_22/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_22/RestoreV2:25*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_22/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_22/RestoreV2:26*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_22/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_22/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_22/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_22/RestoreV2:28*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_22/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_22/RestoreV2:29*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_22/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_22/RestoreV2:30*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_22/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_22/RestoreV2:31*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_22/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_22/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_22/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_22/RestoreV2:33*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_22/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_22/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_22/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_22/RestoreV2:35*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_22/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_22/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_22/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_22/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_22/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_22/RestoreV2:38*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_22/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_22/RestoreV2:39*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_22/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_22/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_22/Assign_41Assignppo_agent/ppo2_model/pi/wsave_22/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_22/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_22/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_22/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_22/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_22/Assign_44Assignppo_agent/ppo2_model/vf/bsave_22/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_22/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_22/RestoreV2:45*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_22/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_22/RestoreV2:46*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_22/Assign_47Assignppo_agent/ppo2_model/vf/wsave_22/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_22/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_22/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_22/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_22/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_22/restore_shardNoOp^save_22/Assign^save_22/Assign_1^save_22/Assign_10^save_22/Assign_11^save_22/Assign_12^save_22/Assign_13^save_22/Assign_14^save_22/Assign_15^save_22/Assign_16^save_22/Assign_17^save_22/Assign_18^save_22/Assign_19^save_22/Assign_2^save_22/Assign_20^save_22/Assign_21^save_22/Assign_22^save_22/Assign_23^save_22/Assign_24^save_22/Assign_25^save_22/Assign_26^save_22/Assign_27^save_22/Assign_28^save_22/Assign_29^save_22/Assign_3^save_22/Assign_30^save_22/Assign_31^save_22/Assign_32^save_22/Assign_33^save_22/Assign_34^save_22/Assign_35^save_22/Assign_36^save_22/Assign_37^save_22/Assign_38^save_22/Assign_39^save_22/Assign_4^save_22/Assign_40^save_22/Assign_41^save_22/Assign_42^save_22/Assign_43^save_22/Assign_44^save_22/Assign_45^save_22/Assign_46^save_22/Assign_47^save_22/Assign_48^save_22/Assign_49^save_22/Assign_5^save_22/Assign_6^save_22/Assign_7^save_22/Assign_8^save_22/Assign_9
3
save_22/restore_allNoOp^save_22/restore_shard
\
save_23/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_23/filenamePlaceholderWithDefaultsave_23/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_23/ConstPlaceholderWithDefaultsave_23/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_23/StringJoin/inputs_1Const*<
value3B1 B+_temp_cf1c3d011cc647f69a5b6d19e3e86c7f/part*
dtype0*
_output_shapes
: 
~
save_23/StringJoin
StringJoinsave_23/Constsave_23/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_23/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_23/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_23/ShardedFilenameShardedFilenamesave_23/StringJoinsave_23/ShardedFilename/shardsave_23/num_shards*
_output_shapes
: 
�
save_23/SaveV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
save_23/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_23/SaveV2SaveV2save_23/ShardedFilenamesave_23/SaveV2/tensor_namessave_23/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_23/control_dependencyIdentitysave_23/ShardedFilename^save_23/SaveV2*
T0**
_class 
loc:@save_23/ShardedFilename*
_output_shapes
: 
�
.save_23/MergeV2Checkpoints/checkpoint_prefixesPacksave_23/ShardedFilename^save_23/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_23/MergeV2CheckpointsMergeV2Checkpoints.save_23/MergeV2Checkpoints/checkpoint_prefixessave_23/Const*
delete_old_dirs(
�
save_23/IdentityIdentitysave_23/Const^save_23/MergeV2Checkpoints^save_23/control_dependency*
T0*
_output_shapes
: 
�
save_23/RestoreV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
"save_23/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_23/RestoreV2	RestoreV2save_23/Constsave_23/RestoreV2/tensor_names"save_23/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_23/AssignAssignbeta1_powersave_23/RestoreV2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_23/Assign_1Assignbeta2_powersave_23/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_23/Assign_2Assignppo_agent/ppo2_model/pi/bsave_23/RestoreV2:2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_23/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_23/RestoreV2:3*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_23/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_23/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_23/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_23/RestoreV2:5*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_23/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_23/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_23/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_23/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_23/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_23/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_23/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_23/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_23/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_23/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_23/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_23/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_23/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_23/RestoreV2:12*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_23/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_23/RestoreV2:13*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_23/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_23/RestoreV2:14*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_23/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_23/RestoreV2:15*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_23/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_23/RestoreV2:16*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_23/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_23/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_23/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_23/RestoreV2:18*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_23/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_23/RestoreV2:19*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_23/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_23/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_23/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_23/RestoreV2:21*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_23/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_23/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_23/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_23/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_23/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_23/RestoreV2:24*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_23/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_23/RestoreV2:25*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_23/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_23/RestoreV2:26*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_23/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_23/RestoreV2:27*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_23/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_23/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_23/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_23/RestoreV2:29*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_23/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_23/RestoreV2:30*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_23/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_23/RestoreV2:31*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_23/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_23/RestoreV2:32*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_23/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_23/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_23/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_23/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_23/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_23/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_23/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_23/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_23/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_23/RestoreV2:37*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_23/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_23/RestoreV2:38*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_23/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_23/RestoreV2:39*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_23/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_23/RestoreV2:40*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_23/Assign_41Assignppo_agent/ppo2_model/pi/wsave_23/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_23/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_23/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_23/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_23/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_23/Assign_44Assignppo_agent/ppo2_model/vf/bsave_23/RestoreV2:44*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_23/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_23/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_23/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_23/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_23/Assign_47Assignppo_agent/ppo2_model/vf/wsave_23/RestoreV2:47*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_23/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_23/RestoreV2:48*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_23/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_23/RestoreV2:49*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_23/restore_shardNoOp^save_23/Assign^save_23/Assign_1^save_23/Assign_10^save_23/Assign_11^save_23/Assign_12^save_23/Assign_13^save_23/Assign_14^save_23/Assign_15^save_23/Assign_16^save_23/Assign_17^save_23/Assign_18^save_23/Assign_19^save_23/Assign_2^save_23/Assign_20^save_23/Assign_21^save_23/Assign_22^save_23/Assign_23^save_23/Assign_24^save_23/Assign_25^save_23/Assign_26^save_23/Assign_27^save_23/Assign_28^save_23/Assign_29^save_23/Assign_3^save_23/Assign_30^save_23/Assign_31^save_23/Assign_32^save_23/Assign_33^save_23/Assign_34^save_23/Assign_35^save_23/Assign_36^save_23/Assign_37^save_23/Assign_38^save_23/Assign_39^save_23/Assign_4^save_23/Assign_40^save_23/Assign_41^save_23/Assign_42^save_23/Assign_43^save_23/Assign_44^save_23/Assign_45^save_23/Assign_46^save_23/Assign_47^save_23/Assign_48^save_23/Assign_49^save_23/Assign_5^save_23/Assign_6^save_23/Assign_7^save_23/Assign_8^save_23/Assign_9
3
save_23/restore_allNoOp^save_23/restore_shard
\
save_24/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_24/filenamePlaceholderWithDefaultsave_24/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_24/ConstPlaceholderWithDefaultsave_24/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_24/StringJoin/inputs_1Const*<
value3B1 B+_temp_04a9535f3b1148549b06ff37119ec695/part*
dtype0*
_output_shapes
: 
~
save_24/StringJoin
StringJoinsave_24/Constsave_24/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_24/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_24/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_24/ShardedFilenameShardedFilenamesave_24/StringJoinsave_24/ShardedFilename/shardsave_24/num_shards*
_output_shapes
: 
�
save_24/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_24/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_24/SaveV2SaveV2save_24/ShardedFilenamesave_24/SaveV2/tensor_namessave_24/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_24/control_dependencyIdentitysave_24/ShardedFilename^save_24/SaveV2*
T0**
_class 
loc:@save_24/ShardedFilename*
_output_shapes
: 
�
.save_24/MergeV2Checkpoints/checkpoint_prefixesPacksave_24/ShardedFilename^save_24/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_24/MergeV2CheckpointsMergeV2Checkpoints.save_24/MergeV2Checkpoints/checkpoint_prefixessave_24/Const*
delete_old_dirs(
�
save_24/IdentityIdentitysave_24/Const^save_24/MergeV2Checkpoints^save_24/control_dependency*
T0*
_output_shapes
: 
�
save_24/RestoreV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
"save_24/RestoreV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_24/RestoreV2	RestoreV2save_24/Constsave_24/RestoreV2/tensor_names"save_24/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_24/AssignAssignbeta1_powersave_24/RestoreV2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_24/Assign_1Assignbeta2_powersave_24/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_24/Assign_2Assignppo_agent/ppo2_model/pi/bsave_24/RestoreV2:2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_24/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_24/RestoreV2:3*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_24/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_24/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_24/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_24/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_24/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_24/RestoreV2:6*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_24/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_24/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_24/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_24/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_24/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_24/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_24/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_24/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_24/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_24/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_24/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_24/RestoreV2:12*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_24/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_24/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_24/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_24/RestoreV2:14*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_24/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_24/RestoreV2:15*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_24/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_24/RestoreV2:16*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_24/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_24/RestoreV2:17*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_24/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_24/RestoreV2:18*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_24/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_24/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_24/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_24/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_24/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_24/RestoreV2:21*
validate_shape(*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*&
_output_shapes
:
�
save_24/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_24/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_24/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_24/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_24/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_24/RestoreV2:24*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_24/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_24/RestoreV2:25*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_24/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_24/RestoreV2:26*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_24/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_24/RestoreV2:27*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_24/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_24/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_24/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_24/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_24/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_24/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_24/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_24/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_24/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_24/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_24/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_24/RestoreV2:33*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_24/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_24/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_24/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_24/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_24/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_24/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_24/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_24/RestoreV2:37*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_24/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_24/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_24/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_24/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_24/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_24/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_24/Assign_41Assignppo_agent/ppo2_model/pi/wsave_24/RestoreV2:41*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_24/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_24/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_24/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_24/RestoreV2:43*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_24/Assign_44Assignppo_agent/ppo2_model/vf/bsave_24/RestoreV2:44*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
_output_shapes
:
�
save_24/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_24/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_24/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_24/RestoreV2:46*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_24/Assign_47Assignppo_agent/ppo2_model/vf/wsave_24/RestoreV2:47*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_24/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_24/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_24/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_24/RestoreV2:49*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_24/restore_shardNoOp^save_24/Assign^save_24/Assign_1^save_24/Assign_10^save_24/Assign_11^save_24/Assign_12^save_24/Assign_13^save_24/Assign_14^save_24/Assign_15^save_24/Assign_16^save_24/Assign_17^save_24/Assign_18^save_24/Assign_19^save_24/Assign_2^save_24/Assign_20^save_24/Assign_21^save_24/Assign_22^save_24/Assign_23^save_24/Assign_24^save_24/Assign_25^save_24/Assign_26^save_24/Assign_27^save_24/Assign_28^save_24/Assign_29^save_24/Assign_3^save_24/Assign_30^save_24/Assign_31^save_24/Assign_32^save_24/Assign_33^save_24/Assign_34^save_24/Assign_35^save_24/Assign_36^save_24/Assign_37^save_24/Assign_38^save_24/Assign_39^save_24/Assign_4^save_24/Assign_40^save_24/Assign_41^save_24/Assign_42^save_24/Assign_43^save_24/Assign_44^save_24/Assign_45^save_24/Assign_46^save_24/Assign_47^save_24/Assign_48^save_24/Assign_49^save_24/Assign_5^save_24/Assign_6^save_24/Assign_7^save_24/Assign_8^save_24/Assign_9
3
save_24/restore_allNoOp^save_24/restore_shard
\
save_25/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_25/filenamePlaceholderWithDefaultsave_25/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_25/ConstPlaceholderWithDefaultsave_25/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_25/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_7a0a9280b19e4e6a94a4795e22aed9dc/part*
_output_shapes
: 
~
save_25/StringJoin
StringJoinsave_25/Constsave_25/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_25/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_25/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_25/ShardedFilenameShardedFilenamesave_25/StringJoinsave_25/ShardedFilename/shardsave_25/num_shards*
_output_shapes
: 
�
save_25/SaveV2/tensor_namesConst*
dtype0*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
_output_shapes
:2
�
save_25/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_25/SaveV2SaveV2save_25/ShardedFilenamesave_25/SaveV2/tensor_namessave_25/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_25/control_dependencyIdentitysave_25/ShardedFilename^save_25/SaveV2*
T0**
_class 
loc:@save_25/ShardedFilename*
_output_shapes
: 
�
.save_25/MergeV2Checkpoints/checkpoint_prefixesPacksave_25/ShardedFilename^save_25/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_25/MergeV2CheckpointsMergeV2Checkpoints.save_25/MergeV2Checkpoints/checkpoint_prefixessave_25/Const*
delete_old_dirs(
�
save_25/IdentityIdentitysave_25/Const^save_25/MergeV2Checkpoints^save_25/control_dependency*
T0*
_output_shapes
: 
�
save_25/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_25/RestoreV2/shape_and_slicesConst*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:2
�
save_25/RestoreV2	RestoreV2save_25/Constsave_25/RestoreV2/tensor_names"save_25/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_25/AssignAssignbeta1_powersave_25/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_25/Assign_1Assignbeta2_powersave_25/RestoreV2:1*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_25/Assign_2Assignppo_agent/ppo2_model/pi/bsave_25/RestoreV2:2*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_25/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_25/RestoreV2:3*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_25/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_25/RestoreV2:4*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
:
�
save_25/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_25/RestoreV2:5*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_25/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_25/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_25/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_25/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_25/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_25/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_25/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_25/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_25/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_25/RestoreV2:10*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_25/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_25/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_25/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_25/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_25/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_25/RestoreV2:13*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_25/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_25/RestoreV2:14*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_25/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_25/RestoreV2:15*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_25/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_25/RestoreV2:16*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*&
_output_shapes
:
�
save_25/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_25/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_25/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_25/RestoreV2:18*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_25/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_25/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_25/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_25/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_25/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_25/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_25/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_25/RestoreV2:22*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_25/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_25/RestoreV2:23*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_25/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_25/RestoreV2:24*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_25/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_25/RestoreV2:25*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_25/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_25/RestoreV2:26*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_25/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_25/RestoreV2:27*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_25/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_25/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_25/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_25/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_25/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_25/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_25/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_25/RestoreV2:31*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_25/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_25/RestoreV2:32*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_25/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_25/RestoreV2:33*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_25/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_25/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_25/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_25/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_25/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_25/RestoreV2:36*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_25/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_25/RestoreV2:37*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_25/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_25/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_25/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_25/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_25/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_25/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_25/Assign_41Assignppo_agent/ppo2_model/pi/wsave_25/RestoreV2:41*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_25/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_25/RestoreV2:42*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_25/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_25/RestoreV2:43*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_25/Assign_44Assignppo_agent/ppo2_model/vf/bsave_25/RestoreV2:44*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_25/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_25/RestoreV2:45*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_25/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_25/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_25/Assign_47Assignppo_agent/ppo2_model/vf/wsave_25/RestoreV2:47*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_25/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_25/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_25/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_25/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_25/restore_shardNoOp^save_25/Assign^save_25/Assign_1^save_25/Assign_10^save_25/Assign_11^save_25/Assign_12^save_25/Assign_13^save_25/Assign_14^save_25/Assign_15^save_25/Assign_16^save_25/Assign_17^save_25/Assign_18^save_25/Assign_19^save_25/Assign_2^save_25/Assign_20^save_25/Assign_21^save_25/Assign_22^save_25/Assign_23^save_25/Assign_24^save_25/Assign_25^save_25/Assign_26^save_25/Assign_27^save_25/Assign_28^save_25/Assign_29^save_25/Assign_3^save_25/Assign_30^save_25/Assign_31^save_25/Assign_32^save_25/Assign_33^save_25/Assign_34^save_25/Assign_35^save_25/Assign_36^save_25/Assign_37^save_25/Assign_38^save_25/Assign_39^save_25/Assign_4^save_25/Assign_40^save_25/Assign_41^save_25/Assign_42^save_25/Assign_43^save_25/Assign_44^save_25/Assign_45^save_25/Assign_46^save_25/Assign_47^save_25/Assign_48^save_25/Assign_49^save_25/Assign_5^save_25/Assign_6^save_25/Assign_7^save_25/Assign_8^save_25/Assign_9
3
save_25/restore_allNoOp^save_25/restore_shard
\
save_26/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_26/filenamePlaceholderWithDefaultsave_26/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_26/ConstPlaceholderWithDefaultsave_26/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_26/StringJoin/inputs_1Const*<
value3B1 B+_temp_fa272f89bef04943b5d26998d4802afe/part*
dtype0*
_output_shapes
: 
~
save_26/StringJoin
StringJoinsave_26/Constsave_26/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_26/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_26/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_26/ShardedFilenameShardedFilenamesave_26/StringJoinsave_26/ShardedFilename/shardsave_26/num_shards*
_output_shapes
: 
�
save_26/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_26/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_26/SaveV2SaveV2save_26/ShardedFilenamesave_26/SaveV2/tensor_namessave_26/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_26/control_dependencyIdentitysave_26/ShardedFilename^save_26/SaveV2*
T0**
_class 
loc:@save_26/ShardedFilename*
_output_shapes
: 
�
.save_26/MergeV2Checkpoints/checkpoint_prefixesPacksave_26/ShardedFilename^save_26/control_dependency*
N*
T0*

axis *
_output_shapes
:
�
save_26/MergeV2CheckpointsMergeV2Checkpoints.save_26/MergeV2Checkpoints/checkpoint_prefixessave_26/Const*
delete_old_dirs(
�
save_26/IdentityIdentitysave_26/Const^save_26/MergeV2Checkpoints^save_26/control_dependency*
T0*
_output_shapes
: 
�
save_26/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_26/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_26/RestoreV2	RestoreV2save_26/Constsave_26/RestoreV2/tensor_names"save_26/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_26/AssignAssignbeta1_powersave_26/RestoreV2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_26/Assign_1Assignbeta2_powersave_26/RestoreV2:1*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_26/Assign_2Assignppo_agent/ppo2_model/pi/bsave_26/RestoreV2:2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_26/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_26/RestoreV2:3*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_26/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_26/RestoreV2:4*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_26/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_26/RestoreV2:5*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_26/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_26/RestoreV2:6*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_26/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_26/RestoreV2:7*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
_output_shapes
:
�
save_26/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_26/RestoreV2:8*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_26/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_26/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_26/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_26/RestoreV2:10*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_26/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_26/RestoreV2:11*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_26/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_26/RestoreV2:12*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
_output_shapes
:
�
save_26/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_26/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_26/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_26/RestoreV2:14*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_26/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_26/RestoreV2:15*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_26/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_26/RestoreV2:16*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*&
_output_shapes
:
�
save_26/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_26/RestoreV2:17*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_26/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_26/RestoreV2:18*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_26/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_26/RestoreV2:19*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
_output_shapes
:
�
save_26/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_26/RestoreV2:20*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_26/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_26/RestoreV2:21*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_26/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_26/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_26/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_26/RestoreV2:23*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_26/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_26/RestoreV2:24*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_26/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_26/RestoreV2:25*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_26/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_26/RestoreV2:26*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_26/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_26/RestoreV2:27*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_26/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_26/RestoreV2:28*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_26/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_26/RestoreV2:29*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_26/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_26/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_26/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_26/RestoreV2:31*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_26/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_26/RestoreV2:32*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_26/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_26/RestoreV2:33*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_26/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_26/RestoreV2:34*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_26/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_26/RestoreV2:35*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_26/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_26/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_26/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_26/RestoreV2:37*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
_output_shapes
:@
�
save_26/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_26/RestoreV2:38*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_26/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_26/RestoreV2:39*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_26/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_26/RestoreV2:40*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_26/Assign_41Assignppo_agent/ppo2_model/pi/wsave_26/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_26/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_26/RestoreV2:42*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
_output_shapes

:@
�
save_26/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_26/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_26/Assign_44Assignppo_agent/ppo2_model/vf/bsave_26/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_26/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_26/RestoreV2:45*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_26/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_26/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_26/Assign_47Assignppo_agent/ppo2_model/vf/wsave_26/RestoreV2:47*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_26/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_26/RestoreV2:48*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_26/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_26/RestoreV2:49*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_26/restore_shardNoOp^save_26/Assign^save_26/Assign_1^save_26/Assign_10^save_26/Assign_11^save_26/Assign_12^save_26/Assign_13^save_26/Assign_14^save_26/Assign_15^save_26/Assign_16^save_26/Assign_17^save_26/Assign_18^save_26/Assign_19^save_26/Assign_2^save_26/Assign_20^save_26/Assign_21^save_26/Assign_22^save_26/Assign_23^save_26/Assign_24^save_26/Assign_25^save_26/Assign_26^save_26/Assign_27^save_26/Assign_28^save_26/Assign_29^save_26/Assign_3^save_26/Assign_30^save_26/Assign_31^save_26/Assign_32^save_26/Assign_33^save_26/Assign_34^save_26/Assign_35^save_26/Assign_36^save_26/Assign_37^save_26/Assign_38^save_26/Assign_39^save_26/Assign_4^save_26/Assign_40^save_26/Assign_41^save_26/Assign_42^save_26/Assign_43^save_26/Assign_44^save_26/Assign_45^save_26/Assign_46^save_26/Assign_47^save_26/Assign_48^save_26/Assign_49^save_26/Assign_5^save_26/Assign_6^save_26/Assign_7^save_26/Assign_8^save_26/Assign_9
3
save_26/restore_allNoOp^save_26/restore_shard
\
save_27/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_27/filenamePlaceholderWithDefaultsave_27/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_27/ConstPlaceholderWithDefaultsave_27/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_27/StringJoin/inputs_1Const*<
value3B1 B+_temp_321d68ea2abc499cb4f8329d99487f98/part*
dtype0*
_output_shapes
: 
~
save_27/StringJoin
StringJoinsave_27/Constsave_27/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_27/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_27/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_27/ShardedFilenameShardedFilenamesave_27/StringJoinsave_27/ShardedFilename/shardsave_27/num_shards*
_output_shapes
: 
�
save_27/SaveV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
save_27/SaveV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_27/SaveV2SaveV2save_27/ShardedFilenamesave_27/SaveV2/tensor_namessave_27/SaveV2/shape_and_slicesbeta1_powerbeta2_powerppo_agent/ppo2_model/pi/bppo_agent/ppo2_model/pi/b/Adam ppo_agent/ppo2_model/pi/b/Adam_1#ppo_agent/ppo2_model/pi/conv_0/bias(ppo_agent/ppo2_model/pi/conv_0/bias/Adam*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_0/kernel*ppo_agent/ppo2_model/pi/conv_0/kernel/Adam,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1#ppo_agent/ppo2_model/pi/conv_1/bias(ppo_agent/ppo2_model/pi/conv_1/bias/Adam*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1%ppo_agent/ppo2_model/pi/conv_1/kernel*ppo_agent/ppo2_model/pi/conv_1/kernel/Adam,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1)ppo_agent/ppo2_model/pi/conv_initial/bias.ppo_agent/ppo2_model/pi/conv_initial/bias/Adam0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1+ppo_agent/ppo2_model/pi/conv_initial/kernel0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1"ppo_agent/ppo2_model/pi/dense/bias'ppo_agent/ppo2_model/pi/dense/bias/Adam)ppo_agent/ppo2_model/pi/dense/bias/Adam_1$ppo_agent/ppo2_model/pi/dense/kernel)ppo_agent/ppo2_model/pi/dense/kernel/Adam+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_1/bias)ppo_agent/ppo2_model/pi/dense_1/bias/Adam+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_1/kernel+ppo_agent/ppo2_model/pi/dense_1/kernel/Adam-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1$ppo_agent/ppo2_model/pi/dense_2/bias)ppo_agent/ppo2_model/pi/dense_2/bias/Adam+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1&ppo_agent/ppo2_model/pi/dense_2/kernel+ppo_agent/ppo2_model/pi/dense_2/kernel/Adam-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1ppo_agent/ppo2_model/pi/wppo_agent/ppo2_model/pi/w/Adam ppo_agent/ppo2_model/pi/w/Adam_1ppo_agent/ppo2_model/vf/bppo_agent/ppo2_model/vf/b/Adam ppo_agent/ppo2_model/vf/b/Adam_1ppo_agent/ppo2_model/vf/wppo_agent/ppo2_model/vf/w/Adam ppo_agent/ppo2_model/vf/w/Adam_1*@
dtypes6
422
�
save_27/control_dependencyIdentitysave_27/ShardedFilename^save_27/SaveV2*
T0**
_class 
loc:@save_27/ShardedFilename*
_output_shapes
: 
�
.save_27/MergeV2Checkpoints/checkpoint_prefixesPacksave_27/ShardedFilename^save_27/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_27/MergeV2CheckpointsMergeV2Checkpoints.save_27/MergeV2Checkpoints/checkpoint_prefixessave_27/Const*
delete_old_dirs(
�
save_27/IdentityIdentitysave_27/Const^save_27/MergeV2Checkpoints^save_27/control_dependency*
T0*
_output_shapes
: 
�
save_27/RestoreV2/tensor_namesConst*�
value�B�2Bbeta1_powerBbeta2_powerBppo_agent/ppo2_model/pi/bBppo_agent/ppo2_model/pi/b/AdamB ppo_agent/ppo2_model/pi/b/Adam_1B#ppo_agent/ppo2_model/pi/conv_0/biasB(ppo_agent/ppo2_model/pi/conv_0/bias/AdamB*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_0/kernelB*ppo_agent/ppo2_model/pi/conv_0/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1B#ppo_agent/ppo2_model/pi/conv_1/biasB(ppo_agent/ppo2_model/pi/conv_1/bias/AdamB*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1B%ppo_agent/ppo2_model/pi/conv_1/kernelB*ppo_agent/ppo2_model/pi/conv_1/kernel/AdamB,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1B)ppo_agent/ppo2_model/pi/conv_initial/biasB.ppo_agent/ppo2_model/pi/conv_initial/bias/AdamB0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1B+ppo_agent/ppo2_model/pi/conv_initial/kernelB0ppo_agent/ppo2_model/pi/conv_initial/kernel/AdamB2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1B"ppo_agent/ppo2_model/pi/dense/biasB'ppo_agent/ppo2_model/pi/dense/bias/AdamB)ppo_agent/ppo2_model/pi/dense/bias/Adam_1B$ppo_agent/ppo2_model/pi/dense/kernelB)ppo_agent/ppo2_model/pi/dense/kernel/AdamB+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_1/biasB)ppo_agent/ppo2_model/pi/dense_1/bias/AdamB+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_1/kernelB+ppo_agent/ppo2_model/pi/dense_1/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1B$ppo_agent/ppo2_model/pi/dense_2/biasB)ppo_agent/ppo2_model/pi/dense_2/bias/AdamB+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1B&ppo_agent/ppo2_model/pi/dense_2/kernelB+ppo_agent/ppo2_model/pi/dense_2/kernel/AdamB-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1Bppo_agent/ppo2_model/pi/wBppo_agent/ppo2_model/pi/w/AdamB ppo_agent/ppo2_model/pi/w/Adam_1Bppo_agent/ppo2_model/vf/bBppo_agent/ppo2_model/vf/b/AdamB ppo_agent/ppo2_model/vf/b/Adam_1Bppo_agent/ppo2_model/vf/wBppo_agent/ppo2_model/vf/w/AdamB ppo_agent/ppo2_model/vf/w/Adam_1*
dtype0*
_output_shapes
:2
�
"save_27/RestoreV2/shape_and_slicesConst*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
�
save_27/RestoreV2	RestoreV2save_27/Constsave_27/RestoreV2/tensor_names"save_27/RestoreV2/shape_and_slices*@
dtypes6
422*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_27/AssignAssignbeta1_powersave_27/RestoreV2*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
_output_shapes
: 
�
save_27/Assign_1Assignbeta2_powersave_27/RestoreV2:1*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
: 
�
save_27/Assign_2Assignppo_agent/ppo2_model/pi/bsave_27/RestoreV2:2*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_27/Assign_3Assignppo_agent/ppo2_model/pi/b/Adamsave_27/RestoreV2:3*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
_output_shapes
:
�
save_27/Assign_4Assign ppo_agent/ppo2_model/pi/b/Adam_1save_27/RestoreV2:4*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_27/Assign_5Assign#ppo_agent/ppo2_model/pi/conv_0/biassave_27/RestoreV2:5*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_27/Assign_6Assign(ppo_agent/ppo2_model/pi/conv_0/bias/Adamsave_27/RestoreV2:6*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_27/Assign_7Assign*ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1save_27/RestoreV2:7*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_0/bias*
_output_shapes
:
�
save_27/Assign_8Assign%ppo_agent/ppo2_model/pi/conv_0/kernelsave_27/RestoreV2:8*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_27/Assign_9Assign*ppo_agent/ppo2_model/pi/conv_0/kernel/Adamsave_27/RestoreV2:9*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*
validate_shape(*&
_output_shapes
:
�
save_27/Assign_10Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1save_27/RestoreV2:10*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_0/kernel*&
_output_shapes
:
�
save_27/Assign_11Assign#ppo_agent/ppo2_model/pi/conv_1/biassave_27/RestoreV2:11*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_27/Assign_12Assign(ppo_agent/ppo2_model/pi/conv_1/bias/Adamsave_27/RestoreV2:12*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_27/Assign_13Assign*ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1save_27/RestoreV2:13*
use_locking(*
T0*6
_class,
*(loc:@ppo_agent/ppo2_model/pi/conv_1/bias*
validate_shape(*
_output_shapes
:
�
save_27/Assign_14Assign%ppo_agent/ppo2_model/pi/conv_1/kernelsave_27/RestoreV2:14*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_27/Assign_15Assign*ppo_agent/ppo2_model/pi/conv_1/kernel/Adamsave_27/RestoreV2:15*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_27/Assign_16Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1save_27/RestoreV2:16*
T0*8
_class.
,*loc:@ppo_agent/ppo2_model/pi/conv_1/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_27/Assign_17Assign)ppo_agent/ppo2_model/pi/conv_initial/biassave_27/RestoreV2:17*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_27/Assign_18Assign.ppo_agent/ppo2_model/pi/conv_initial/bias/Adamsave_27/RestoreV2:18*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_27/Assign_19Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1save_27/RestoreV2:19*
use_locking(*
T0*<
_class2
0.loc:@ppo_agent/ppo2_model/pi/conv_initial/bias*
validate_shape(*
_output_shapes
:
�
save_27/Assign_20Assign+ppo_agent/ppo2_model/pi/conv_initial/kernelsave_27/RestoreV2:20*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_27/Assign_21Assign0ppo_agent/ppo2_model/pi/conv_initial/kernel/Adamsave_27/RestoreV2:21*
use_locking(*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*&
_output_shapes
:
�
save_27/Assign_22Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1save_27/RestoreV2:22*
T0*>
_class4
20loc:@ppo_agent/ppo2_model/pi/conv_initial/kernel*
validate_shape(*
use_locking(*&
_output_shapes
:
�
save_27/Assign_23Assign"ppo_agent/ppo2_model/pi/dense/biassave_27/RestoreV2:23*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_27/Assign_24Assign'ppo_agent/ppo2_model/pi/dense/bias/Adamsave_27/RestoreV2:24*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
_output_shapes
:@
�
save_27/Assign_25Assign)ppo_agent/ppo2_model/pi/dense/bias/Adam_1save_27/RestoreV2:25*
use_locking(*
T0*5
_class+
)'loc:@ppo_agent/ppo2_model/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_27/Assign_26Assign$ppo_agent/ppo2_model/pi/dense/kernelsave_27/RestoreV2:26*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
_output_shapes
:	�@
�
save_27/Assign_27Assign)ppo_agent/ppo2_model/pi/dense/kernel/Adamsave_27/RestoreV2:27*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@
�
save_27/Assign_28Assign+ppo_agent/ppo2_model/pi/dense/kernel/Adam_1save_27/RestoreV2:28*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_27/Assign_29Assign$ppo_agent/ppo2_model/pi/dense_1/biassave_27/RestoreV2:29*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_27/Assign_30Assign)ppo_agent/ppo2_model/pi/dense_1/bias/Adamsave_27/RestoreV2:30*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_27/Assign_31Assign+ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1save_27/RestoreV2:31*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_1/bias*
_output_shapes
:@
�
save_27/Assign_32Assign&ppo_agent/ppo2_model/pi/dense_1/kernelsave_27/RestoreV2:32*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_27/Assign_33Assign+ppo_agent/ppo2_model/pi/dense_1/kernel/Adamsave_27/RestoreV2:33*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
_output_shapes

:@@
�
save_27/Assign_34Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1save_27/RestoreV2:34*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_27/Assign_35Assign$ppo_agent/ppo2_model/pi/dense_2/biassave_27/RestoreV2:35*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_27/Assign_36Assign)ppo_agent/ppo2_model/pi/dense_2/bias/Adamsave_27/RestoreV2:36*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_27/Assign_37Assign+ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1save_27/RestoreV2:37*
validate_shape(*
use_locking(*
T0*7
_class-
+)loc:@ppo_agent/ppo2_model/pi/dense_2/bias*
_output_shapes
:@
�
save_27/Assign_38Assign&ppo_agent/ppo2_model/pi/dense_2/kernelsave_27/RestoreV2:38*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@@
�
save_27/Assign_39Assign+ppo_agent/ppo2_model/pi/dense_2/kernel/Adamsave_27/RestoreV2:39*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
_output_shapes

:@@
�
save_27/Assign_40Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1save_27/RestoreV2:40*
T0*9
_class/
-+loc:@ppo_agent/ppo2_model/pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:@@
�
save_27/Assign_41Assignppo_agent/ppo2_model/pi/wsave_27/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_27/Assign_42Assignppo_agent/ppo2_model/pi/w/Adamsave_27/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_27/Assign_43Assign ppo_agent/ppo2_model/pi/w/Adam_1save_27/RestoreV2:43*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/pi/w*
validate_shape(*
_output_shapes

:@
�
save_27/Assign_44Assignppo_agent/ppo2_model/vf/bsave_27/RestoreV2:44*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_27/Assign_45Assignppo_agent/ppo2_model/vf/b/Adamsave_27/RestoreV2:45*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_27/Assign_46Assign ppo_agent/ppo2_model/vf/b/Adam_1save_27/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/b*
validate_shape(*
_output_shapes
:
�
save_27/Assign_47Assignppo_agent/ppo2_model/vf/wsave_27/RestoreV2:47*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
use_locking(*
_output_shapes

:@
�
save_27/Assign_48Assignppo_agent/ppo2_model/vf/w/Adamsave_27/RestoreV2:48*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
validate_shape(*
_output_shapes

:@
�
save_27/Assign_49Assign ppo_agent/ppo2_model/vf/w/Adam_1save_27/RestoreV2:49*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@ppo_agent/ppo2_model/vf/w*
_output_shapes

:@
�
save_27/restore_shardNoOp^save_27/Assign^save_27/Assign_1^save_27/Assign_10^save_27/Assign_11^save_27/Assign_12^save_27/Assign_13^save_27/Assign_14^save_27/Assign_15^save_27/Assign_16^save_27/Assign_17^save_27/Assign_18^save_27/Assign_19^save_27/Assign_2^save_27/Assign_20^save_27/Assign_21^save_27/Assign_22^save_27/Assign_23^save_27/Assign_24^save_27/Assign_25^save_27/Assign_26^save_27/Assign_27^save_27/Assign_28^save_27/Assign_29^save_27/Assign_3^save_27/Assign_30^save_27/Assign_31^save_27/Assign_32^save_27/Assign_33^save_27/Assign_34^save_27/Assign_35^save_27/Assign_36^save_27/Assign_37^save_27/Assign_38^save_27/Assign_39^save_27/Assign_4^save_27/Assign_40^save_27/Assign_41^save_27/Assign_42^save_27/Assign_43^save_27/Assign_44^save_27/Assign_45^save_27/Assign_46^save_27/Assign_47^save_27/Assign_48^save_27/Assign_49^save_27/Assign_5^save_27/Assign_6^save_27/Assign_7^save_27/Assign_8^save_27/Assign_9
3
save_27/restore_allNoOp^save_27/restore_shard "E
save_27/Const:0save_27/Identity:0save_27/restore_all (5 @F8"�M
	variables�L�L
�
-ppo_agent/ppo2_model/pi/conv_initial/kernel:02ppo_agent/ppo2_model/pi/conv_initial/kernel/Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/read:02Hppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform:08
�
+ppo_agent/ppo2_model/pi/conv_initial/bias:00ppo_agent/ppo2_model/pi/conv_initial/bias/Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/read:02=ppo_agent/ppo2_model/pi/conv_initial/bias/Initializer/zeros:08
�
'ppo_agent/ppo2_model/pi/conv_0/kernel:0,ppo_agent/ppo2_model/pi/conv_0/kernel/Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/read:02Bppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform:08
�
%ppo_agent/ppo2_model/pi/conv_0/bias:0*ppo_agent/ppo2_model/pi/conv_0/bias/Assign*ppo_agent/ppo2_model/pi/conv_0/bias/read:027ppo_agent/ppo2_model/pi/conv_0/bias/Initializer/zeros:08
�
'ppo_agent/ppo2_model/pi/conv_1/kernel:0,ppo_agent/ppo2_model/pi/conv_1/kernel/Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/read:02Bppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform:08
�
%ppo_agent/ppo2_model/pi/conv_1/bias:0*ppo_agent/ppo2_model/pi/conv_1/bias/Assign*ppo_agent/ppo2_model/pi/conv_1/bias/read:027ppo_agent/ppo2_model/pi/conv_1/bias/Initializer/zeros:08
�
&ppo_agent/ppo2_model/pi/dense/kernel:0+ppo_agent/ppo2_model/pi/dense/kernel/Assign+ppo_agent/ppo2_model/pi/dense/kernel/read:02Appo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform:08
�
$ppo_agent/ppo2_model/pi/dense/bias:0)ppo_agent/ppo2_model/pi/dense/bias/Assign)ppo_agent/ppo2_model/pi/dense/bias/read:026ppo_agent/ppo2_model/pi/dense/bias/Initializer/zeros:08
�
(ppo_agent/ppo2_model/pi/dense_1/kernel:0-ppo_agent/ppo2_model/pi/dense_1/kernel/Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/read:02Cppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform:08
�
&ppo_agent/ppo2_model/pi/dense_1/bias:0+ppo_agent/ppo2_model/pi/dense_1/bias/Assign+ppo_agent/ppo2_model/pi/dense_1/bias/read:028ppo_agent/ppo2_model/pi/dense_1/bias/Initializer/zeros:08
�
(ppo_agent/ppo2_model/pi/dense_2/kernel:0-ppo_agent/ppo2_model/pi/dense_2/kernel/Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/read:02Cppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform:08
�
&ppo_agent/ppo2_model/pi/dense_2/bias:0+ppo_agent/ppo2_model/pi/dense_2/bias/Assign+ppo_agent/ppo2_model/pi/dense_2/bias/read:028ppo_agent/ppo2_model/pi/dense_2/bias/Initializer/zeros:08
�
ppo_agent/ppo2_model/pi/w:0 ppo_agent/ppo2_model/pi/w/Assign ppo_agent/ppo2_model/pi/w/read:025ppo_agent/ppo2_model/pi/w/Initializer/initial_value:08
�
ppo_agent/ppo2_model/pi/b:0 ppo_agent/ppo2_model/pi/b/Assign ppo_agent/ppo2_model/pi/b/read:02-ppo_agent/ppo2_model/pi/b/Initializer/Const:08
�
ppo_agent/ppo2_model/vf/w:0 ppo_agent/ppo2_model/vf/w/Assign ppo_agent/ppo2_model/vf/w/read:025ppo_agent/ppo2_model/vf/w/Initializer/initial_value:08
�
ppo_agent/ppo2_model/vf/b:0 ppo_agent/ppo2_model/vf/b/Assign ppo_agent/ppo2_model/vf/b/read:02-ppo_agent/ppo2_model/vf/b/Initializer/Const:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
�
2ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam:07ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Assign7ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/read:02Dppo_agent/ppo2_model/pi/conv_initial/kernel/Adam/Initializer/zeros:0
�
4ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1:09ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Assign9ppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/read:02Fppo_agent/ppo2_model/pi/conv_initial/kernel/Adam_1/Initializer/zeros:0
�
0ppo_agent/ppo2_model/pi/conv_initial/bias/Adam:05ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/Assign5ppo_agent/ppo2_model/pi/conv_initial/bias/Adam/read:02Bppo_agent/ppo2_model/pi/conv_initial/bias/Adam/Initializer/zeros:0
�
2ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1:07ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/Assign7ppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/read:02Dppo_agent/ppo2_model/pi/conv_initial/bias/Adam_1/Initializer/zeros:0
�
,ppo_agent/ppo2_model/pi/conv_0/kernel/Adam:01ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Assign1ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/read:02>ppo_agent/ppo2_model/pi/conv_0/kernel/Adam/Initializer/zeros:0
�
.ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1:03ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Assign3ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/read:02@ppo_agent/ppo2_model/pi/conv_0/kernel/Adam_1/Initializer/zeros:0
�
*ppo_agent/ppo2_model/pi/conv_0/bias/Adam:0/ppo_agent/ppo2_model/pi/conv_0/bias/Adam/Assign/ppo_agent/ppo2_model/pi/conv_0/bias/Adam/read:02<ppo_agent/ppo2_model/pi/conv_0/bias/Adam/Initializer/zeros:0
�
,ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1:01ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/Assign1ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/read:02>ppo_agent/ppo2_model/pi/conv_0/bias/Adam_1/Initializer/zeros:0
�
,ppo_agent/ppo2_model/pi/conv_1/kernel/Adam:01ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Assign1ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/read:02>ppo_agent/ppo2_model/pi/conv_1/kernel/Adam/Initializer/zeros:0
�
.ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1:03ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Assign3ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/read:02@ppo_agent/ppo2_model/pi/conv_1/kernel/Adam_1/Initializer/zeros:0
�
*ppo_agent/ppo2_model/pi/conv_1/bias/Adam:0/ppo_agent/ppo2_model/pi/conv_1/bias/Adam/Assign/ppo_agent/ppo2_model/pi/conv_1/bias/Adam/read:02<ppo_agent/ppo2_model/pi/conv_1/bias/Adam/Initializer/zeros:0
�
,ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1:01ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/Assign1ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/read:02>ppo_agent/ppo2_model/pi/conv_1/bias/Adam_1/Initializer/zeros:0
�
+ppo_agent/ppo2_model/pi/dense/kernel/Adam:00ppo_agent/ppo2_model/pi/dense/kernel/Adam/Assign0ppo_agent/ppo2_model/pi/dense/kernel/Adam/read:02=ppo_agent/ppo2_model/pi/dense/kernel/Adam/Initializer/zeros:0
�
-ppo_agent/ppo2_model/pi/dense/kernel/Adam_1:02ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Assign2ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/read:02?ppo_agent/ppo2_model/pi/dense/kernel/Adam_1/Initializer/zeros:0
�
)ppo_agent/ppo2_model/pi/dense/bias/Adam:0.ppo_agent/ppo2_model/pi/dense/bias/Adam/Assign.ppo_agent/ppo2_model/pi/dense/bias/Adam/read:02;ppo_agent/ppo2_model/pi/dense/bias/Adam/Initializer/zeros:0
�
+ppo_agent/ppo2_model/pi/dense/bias/Adam_1:00ppo_agent/ppo2_model/pi/dense/bias/Adam_1/Assign0ppo_agent/ppo2_model/pi/dense/bias/Adam_1/read:02=ppo_agent/ppo2_model/pi/dense/bias/Adam_1/Initializer/zeros:0
�
-ppo_agent/ppo2_model/pi/dense_1/kernel/Adam:02ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Assign2ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/read:02?ppo_agent/ppo2_model/pi/dense_1/kernel/Adam/Initializer/zeros:0
�
/ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1:04ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Assign4ppo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/read:02Appo_agent/ppo2_model/pi/dense_1/kernel/Adam_1/Initializer/zeros:0
�
+ppo_agent/ppo2_model/pi/dense_1/bias/Adam:00ppo_agent/ppo2_model/pi/dense_1/bias/Adam/Assign0ppo_agent/ppo2_model/pi/dense_1/bias/Adam/read:02=ppo_agent/ppo2_model/pi/dense_1/bias/Adam/Initializer/zeros:0
�
-ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1:02ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/Assign2ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/read:02?ppo_agent/ppo2_model/pi/dense_1/bias/Adam_1/Initializer/zeros:0
�
-ppo_agent/ppo2_model/pi/dense_2/kernel/Adam:02ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Assign2ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/read:02?ppo_agent/ppo2_model/pi/dense_2/kernel/Adam/Initializer/zeros:0
�
/ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1:04ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Assign4ppo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/read:02Appo_agent/ppo2_model/pi/dense_2/kernel/Adam_1/Initializer/zeros:0
�
+ppo_agent/ppo2_model/pi/dense_2/bias/Adam:00ppo_agent/ppo2_model/pi/dense_2/bias/Adam/Assign0ppo_agent/ppo2_model/pi/dense_2/bias/Adam/read:02=ppo_agent/ppo2_model/pi/dense_2/bias/Adam/Initializer/zeros:0
�
-ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1:02ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/Assign2ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/read:02?ppo_agent/ppo2_model/pi/dense_2/bias/Adam_1/Initializer/zeros:0
�
 ppo_agent/ppo2_model/pi/w/Adam:0%ppo_agent/ppo2_model/pi/w/Adam/Assign%ppo_agent/ppo2_model/pi/w/Adam/read:022ppo_agent/ppo2_model/pi/w/Adam/Initializer/zeros:0
�
"ppo_agent/ppo2_model/pi/w/Adam_1:0'ppo_agent/ppo2_model/pi/w/Adam_1/Assign'ppo_agent/ppo2_model/pi/w/Adam_1/read:024ppo_agent/ppo2_model/pi/w/Adam_1/Initializer/zeros:0
�
 ppo_agent/ppo2_model/pi/b/Adam:0%ppo_agent/ppo2_model/pi/b/Adam/Assign%ppo_agent/ppo2_model/pi/b/Adam/read:022ppo_agent/ppo2_model/pi/b/Adam/Initializer/zeros:0
�
"ppo_agent/ppo2_model/pi/b/Adam_1:0'ppo_agent/ppo2_model/pi/b/Adam_1/Assign'ppo_agent/ppo2_model/pi/b/Adam_1/read:024ppo_agent/ppo2_model/pi/b/Adam_1/Initializer/zeros:0
�
 ppo_agent/ppo2_model/vf/w/Adam:0%ppo_agent/ppo2_model/vf/w/Adam/Assign%ppo_agent/ppo2_model/vf/w/Adam/read:022ppo_agent/ppo2_model/vf/w/Adam/Initializer/zeros:0
�
"ppo_agent/ppo2_model/vf/w/Adam_1:0'ppo_agent/ppo2_model/vf/w/Adam_1/Assign'ppo_agent/ppo2_model/vf/w/Adam_1/read:024ppo_agent/ppo2_model/vf/w/Adam_1/Initializer/zeros:0
�
 ppo_agent/ppo2_model/vf/b/Adam:0%ppo_agent/ppo2_model/vf/b/Adam/Assign%ppo_agent/ppo2_model/vf/b/Adam/read:022ppo_agent/ppo2_model/vf/b/Adam/Initializer/zeros:0
�
"ppo_agent/ppo2_model/vf/b/Adam_1:0'ppo_agent/ppo2_model/vf/b/Adam_1/Assign'ppo_agent/ppo2_model/vf/b/Adam_1/read:024ppo_agent/ppo2_model/vf/b/Adam_1/Initializer/zeros:0"�
trainable_variables��
�
-ppo_agent/ppo2_model/pi/conv_initial/kernel:02ppo_agent/ppo2_model/pi/conv_initial/kernel/Assign2ppo_agent/ppo2_model/pi/conv_initial/kernel/read:02Hppo_agent/ppo2_model/pi/conv_initial/kernel/Initializer/random_uniform:08
�
+ppo_agent/ppo2_model/pi/conv_initial/bias:00ppo_agent/ppo2_model/pi/conv_initial/bias/Assign0ppo_agent/ppo2_model/pi/conv_initial/bias/read:02=ppo_agent/ppo2_model/pi/conv_initial/bias/Initializer/zeros:08
�
'ppo_agent/ppo2_model/pi/conv_0/kernel:0,ppo_agent/ppo2_model/pi/conv_0/kernel/Assign,ppo_agent/ppo2_model/pi/conv_0/kernel/read:02Bppo_agent/ppo2_model/pi/conv_0/kernel/Initializer/random_uniform:08
�
%ppo_agent/ppo2_model/pi/conv_0/bias:0*ppo_agent/ppo2_model/pi/conv_0/bias/Assign*ppo_agent/ppo2_model/pi/conv_0/bias/read:027ppo_agent/ppo2_model/pi/conv_0/bias/Initializer/zeros:08
�
'ppo_agent/ppo2_model/pi/conv_1/kernel:0,ppo_agent/ppo2_model/pi/conv_1/kernel/Assign,ppo_agent/ppo2_model/pi/conv_1/kernel/read:02Bppo_agent/ppo2_model/pi/conv_1/kernel/Initializer/random_uniform:08
�
%ppo_agent/ppo2_model/pi/conv_1/bias:0*ppo_agent/ppo2_model/pi/conv_1/bias/Assign*ppo_agent/ppo2_model/pi/conv_1/bias/read:027ppo_agent/ppo2_model/pi/conv_1/bias/Initializer/zeros:08
�
&ppo_agent/ppo2_model/pi/dense/kernel:0+ppo_agent/ppo2_model/pi/dense/kernel/Assign+ppo_agent/ppo2_model/pi/dense/kernel/read:02Appo_agent/ppo2_model/pi/dense/kernel/Initializer/random_uniform:08
�
$ppo_agent/ppo2_model/pi/dense/bias:0)ppo_agent/ppo2_model/pi/dense/bias/Assign)ppo_agent/ppo2_model/pi/dense/bias/read:026ppo_agent/ppo2_model/pi/dense/bias/Initializer/zeros:08
�
(ppo_agent/ppo2_model/pi/dense_1/kernel:0-ppo_agent/ppo2_model/pi/dense_1/kernel/Assign-ppo_agent/ppo2_model/pi/dense_1/kernel/read:02Cppo_agent/ppo2_model/pi/dense_1/kernel/Initializer/random_uniform:08
�
&ppo_agent/ppo2_model/pi/dense_1/bias:0+ppo_agent/ppo2_model/pi/dense_1/bias/Assign+ppo_agent/ppo2_model/pi/dense_1/bias/read:028ppo_agent/ppo2_model/pi/dense_1/bias/Initializer/zeros:08
�
(ppo_agent/ppo2_model/pi/dense_2/kernel:0-ppo_agent/ppo2_model/pi/dense_2/kernel/Assign-ppo_agent/ppo2_model/pi/dense_2/kernel/read:02Cppo_agent/ppo2_model/pi/dense_2/kernel/Initializer/random_uniform:08
�
&ppo_agent/ppo2_model/pi/dense_2/bias:0+ppo_agent/ppo2_model/pi/dense_2/bias/Assign+ppo_agent/ppo2_model/pi/dense_2/bias/read:028ppo_agent/ppo2_model/pi/dense_2/bias/Initializer/zeros:08
�
ppo_agent/ppo2_model/pi/w:0 ppo_agent/ppo2_model/pi/w/Assign ppo_agent/ppo2_model/pi/w/read:025ppo_agent/ppo2_model/pi/w/Initializer/initial_value:08
�
ppo_agent/ppo2_model/pi/b:0 ppo_agent/ppo2_model/pi/b/Assign ppo_agent/ppo2_model/pi/b/read:02-ppo_agent/ppo2_model/pi/b/Initializer/Const:08
�
ppo_agent/ppo2_model/vf/w:0 ppo_agent/ppo2_model/vf/w/Assign ppo_agent/ppo2_model/vf/w/read:025ppo_agent/ppo2_model/vf/w/Initializer/initial_value:08
�
ppo_agent/ppo2_model/vf/b:0 ppo_agent/ppo2_model/vf/b/Assign ppo_agent/ppo2_model/vf/b/read:02-ppo_agent/ppo2_model/vf/b/Initializer/Const:08"
train_op

Adam*�
serving_default�
6
obs/
ppo_agent/ppo2_model/Ob:01
action'
ppo_agent/ppo2_model/action:0	/
value&
ppo_agent/ppo2_model/value:0A
action_probs1
#ppo_agent/ppo2_model/action_probs:0tensorflow/serving/predict