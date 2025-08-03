---
layout: post
title: "Adding Costmaps to a Custom Cost Function"
tags:
- How-to Guides
- Texture Helper API
- Cost Function
description: "A simple how-to for integrating maps into your cost function"
author: Bogdan Vlahov
---
One common Cost Function component is having a map-based cost.
In our library, we have included a helper class to allow for fast querying on the GPU and equivalent querying on the CPU.
In this post, I will dicuss how to use our TextureHelper API to add maps to your own custom Cost Function.
This post is based on this following [Github Issue](https://github.com/ACDSLab/MPPI-Generic/issues/8) we received and should hopefully go into more depth for curious readers.

{% capture map-cost-def %}
{% include code/map_cost/map_cost.cuh %}
{% endcapture %}
{% assign map-def-snippets = map-cost-def | newline_to_br | strip_newlines | split: '<br />' %}

{% capture map-cost-src %}
{% include code/map_cost/map_cost.cu %}
{% endcapture %}
{% assign map-src-snippets = map-cost-src | newline_to_br | strip_newlines | split: '<br />' %}

## Creating the Cost Function
Let us first set up a minimal example.
Say we have an existing map of safe regions that we then want to use as our cost for a driving robot.
At every cell of the map, the value is the distance (in meters) to the nearest unsafe cell; cells containing obstacles will have a value of zero.
We want our cost to be zero if the robot is more than 2 meters away from an obstacle and an linearly increasing cost as
we get closer to an obstacle.
For the purposes of this example, I will use the 2D Texture Helper but there is also a 3D Texture Helper for 3D maps.
The TextureHelper class defines the map location, scale, orientation, and can hold multiple maps of different scales.
It computes bilinear interpolation between grid cells by default but can be configured to return the nearest grid cell if desred.
The Texture Helper API is also templated on the data type of the map.
We will use `float` in our example to take advantage of linear interpolation but there is also the option of
`char`, `int`, `short`, `long`.
The map queries work on both the CPU and GPU with the GPU queries using the [CUDA Texture API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)
for faster reads.
These classes are defined [here](https://github.com/ACDSLab/MPPI-Generic/tree/main/include/mppi/utils/texture_helpers)
in the MPPI-Generic source code for those wanting to see the implementation details.

### Cost Function Class Header File
The first step is adding the 2D Texture Helper to your custom Cost Function class in the header file.
For this example, I will call my final cost function `MapCost`.
I create a `MapCostImpl` class so that this class can be extended in the future with CRTP compliance.
In the header file shown in Lst. [1](#code1), we set up inheriting from the base Cost class, declare the necessary methods,
and add a `TwoDTextureHelper` class variable and a method to access it (`getTextureHelper()`).
We do not use `std::shared_ptr` for the `TwoDTextureHelper` as the CPU and GPU have different memory locations and allocation methods
which makes `std::shared_ptr` incompatible with CUDA.
It is possible to use a `std::shared_ptr` on the CPU side only and a raw pointer on the GPU side which I shall leave for a future post.
It is important to note that we are overriding `bindToStream()`, `GPUSetup()`,`freeCudaMem()`, and
`paramsToDevice()` as well as the regular overwritten methods.
Towards the bottom, the class `MapCost` is defined as an instantiated form of
`MapCostImpl` that is used as the cost function in MPPI but can't be inherited from due to the lack of
`CLASS_T` templating.
Finally, we have the inclusion of the `map_cost.cu` file where the definitions of the methods are actually stored.
We wrap it in a check to see if the compiler is `nvcc` as our current attempt to allow compilation of a MPPI controller
into a shared library that can then be linked to by a pure C++ object without knowledge of CUDA methods.

{% highlight cuda linenos %}
{% for line in map-def-snippets offset:8 %}{{ line }}
{% endfor %}
{% endhighlight %}
*Listing <a id="code1">1</a>: Header file for the Custom Cost Function using the TextureHelper API*
{: class="codecaption"}

### Cost Function Source File
Next, we go to the definitions of the various methods, starting with the constructor and destructor.
We first note that in the constructor, we create a new `TwoDTextureHelper` variable.
The number passed in indicates how many maps we want to store in the helper (more can be added later using the `addNewTexture()` method).
For this example, we only need a single map but set this higher in cases where you have more maps you want to query.
We set up the destructor to delete the texture helper to prevent CPU memory leaks.

{% highlight cuda %}
{% for line in map-src-snippets offset:1 limit: 11 %}{{ line }}
{% endfor %}
{% endhighlight %}

From there, we go to setting up the methods that interact with the GPU.
The first method is `GPUSetup()` which allocates the GPU memory for the cost function and the `TwoDTextureHelper`.
The final line also ensures that the GPU version of the cost function points to the GPU version of the `TwoDTextureHelper`.
The next three methods, `freeCudaMem()`, `paramsToDevice()` and `bindToStream()`, ensure that the texture helper's equivalent methods are also
called alongside the cost function's.
With this, the texture helper will now update its GPU component whenever the Cost Function updates its GPU component and
does not require separate management.
{% highlight cuda %}
{% for line in map-src-snippets offset:13 limit: 36 %}{{ line }}
{% endfor %}
{% endhighlight %}

We now move on to actually using the map to compute costs.
The CPU and GPU versions of `computeStateCost()` look the same other than the use of `s()` to index into the state on
the CPU and `s[]` on the GPU.
We create a variable `map_index` for clarity on the Texture Helper method inputs.
As there is only have a single map from our constructor in this example, `map_index` is just set to 0.
The first step is checking if the map is ready to be used with  `checkTextureUse(map_index)`.
This prevents us from reading potentially uninitialized memory.
Next, we create the query point that defines where we want to read the map at.
For our simple case, we only need to query using the *x* and *y* positions of the driving robot.
We make use of our macro `O_IND_CLASS()` to allow us to query the state array at the
`POS_X` and `POS_Y` locations as defined in the Dynamics' Parameter structure.
This allows this cost function to be compatible with any Dynamics that defines `POS_X` and `POS_Y` as part of their
output.
We then query the map using `queryTextureAtWorldPose()`.
This method first converts our *x* and *y* position in meters
into the equivalent pixel coordinate of the map and returns the value at that map location, interpolating if
the pixel coordinates are in between grid cells.
With this, we have the distance of our robot from the nearest obstacle and we can set up our cost.
The cost is zero when we are more than 2 meters away from an obstacle and then linearly increases up to a max
cost of two when we are inside an obstacle.
{% highlight cuda linenos %}
{% for line in map-src-snippets offset:50 limit: 36 %}{{ line }}
{% endfor %}
{% endhighlight %}

The full definition file `map_cost.cu` is shown below in Lst. [2](#code2). For our simple example, we set the terminal cost to be zero.
{% highlight cuda linenos %}
{{ map-cost-src }}
{% endhighlight %}
*Listing <a id="code2">2</a>: Source file for the Custom Cost Function using the TextureHelper API*
{: class="codecaption"}

Our cost function is now fully defined and can query a map on both the CPU and GPU.
We shall now go on to discuss the Texture Helper API for filling in the map.
## Filling in the Map

In order to fill in the map n the texture helper, we  need to have a map. we will assume it is stored in a `std::vector<float>` in row-major order and have some number of rows and columns. We also need to have access to the texture helper inside the cost function.
```cuda
using COST_T = MapCost<DubinsDynamics>;
COST_T cost; // Create an instantiation of the cost function
auto tex_helper_ptr = cost.getTexHelper();

std::vector<float> map_data;
int num_rows = 12;
int num_cols = 15;
```
We shall assume that `map_data` is filled in from some outside source like an image or maybe from a ROS topic. Other required information is the resolution of the map. We shall assume that $1$ pixel length is $1\text{m}$ for simplicity and fill it in as follows:
```cuda
float resolution_meters_per_pixel = 1.0;
tex_helper_ptr->updateResolution(map_index, resolution_meters_per_pixel);
```
where `map_index = 0` for our example of only having a single map.
### Map Dimensions
Next, we shall update the texture helper to have the map dimensions. We use the `cudaExtent` structure to store the dimensions as it explicitly labels the dimensions as `width`, `height` and `depth`. We can fill the structure in the following ways.
```cuda
// Method 1: Invididually set dimensions
cudaExtent map_dim;
map_dim.width = num_cols;
map_dim.height = num_rows;
map_dim.depth = 0; // 2D maps have 0 depth

// Method 2: set multiple dimensions at once
map_dim = make_cudaExtent(num_cols, num_rows, 0);
```
Once filled, we pass that to the texture helper using the `setExtent()` method.
```cuda
tex_helper_ptr->setExtent(map_index, map_dim);
```

### Map Rotation and Origin
The map origin and rotation matrix can be tricky to get right as there is the chance to have coordinate frame confusion. The map origin vector and rotation matrix are used to convert the robot state coordinate frame into the map data coordinate frame. The map is queried in image space coordinates where the origin starts in the top left and positve *x* goes right, positive *y* goes down the page, and positive *z* goes out of the page towards the reader (*z* is ignored in the case of a 2D map). This coordinate system stems from the data being stored in a 2D row-major structure.
```c++
   |  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
---+---------------------------------------------
 0 |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 1 |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 2 |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 3 |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 4 |  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0
 5 |  0  1  1  1  0  0  0  0  0  0  0  0  0  0  0
 6 |  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0
 7 |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 8 |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 9 |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
10 |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
11 |  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
```
*Map showing an obstacle located at $x=5,y=2$ in the image coordinate frame*

This can cause problems depending on how the map was constructed. Let us say that there is an obstacle at $x=5,y=2$. If the map is constructed in image space coordinates, then the obstacle location in the map would be as shown in Fig. 1 and queryable at `m[2][5]`. However, most robots operate in North-East-Down (NED) or North-West-Up (NWU) coordinate frames. If the map was constructed in the NWU coordinate frame, an obstacle at $x=5, y=2$ looks like Fig. 2.

```c++
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 | 11
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 | 10
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 |  9
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 |  8
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 |  7
 0  0  0  0  0  0  0  0  0  0  0  0  1  0  0 |  6
 0  0  0  0  0  0  0  0  0  0  0  1  1  1  0 |  5
 0  0  0  0  0  0  0  0  0  0  0  0  1  0  0 |  4
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 |  3
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 |  2
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 |  1
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 |  0
---------------------------------------------+---
14 13 12 11 10  9  8  7  6  5  4  3  2  1  0 |
```
*Map showing an obstacle located at $x=5,y=2$ in the NWU coordinate frame. Notice that array location of the obstacle is no longer at `m[2][5]` as before but rather `m[6][12]` (it is in the 13th column, and the 7th row)*

While the location of the obstacle is the same, the location of where it is stored in the map data has changed. This difference is addressed by adjusting the origin and rotation of the texture helper map. The coordinate frame transformation equation is

$$
\begin{align}
\begin{bmatrix}
x^{image} \\
y^{image} \\
z^{image} \\
\end{bmatrix} = R_{coord} \PP{
\begin{bmatrix}
x^{robot} \\
y^{robot} \\
z^{robot}
\end{bmatrix}
- \begin{bmatrix}
x^{o} \\
y^{o} \\
z^{o}
\end{bmatrix}}. \label{eq:coord_transform}
\end{align}
$$

where $[x^{o}, y^{o}, z^{o}]^\top$ the origin vector and $R_{coord}$ is the rotation matrix. The texture helper origin is the location of the top left corner of the map - `m[0][0]` - in world frame. In the case of  a NWU coordinate frame and our current map dimensions, the texture helper origin is going to be at $x_o = 14, y_o = 24$, one off of the width and height of the map.
```cuda
float3 map_top_left_pos = make_float3(14, 24, 0);
tex_helper_ptr->updateOrigin(map_index, map_top_left_pos);
```
Next is the rotation matrix, $R_{coord}$. We need to go from NWU to image space coordinates. Using the two figures above for guidance, we can see that the vertical component is $x$ in NWU and becomes $y$ in the image coordinate frame. The positive direction in both frames is also reversed so we also need a negation. We can also see that $y$ in NWU is equivalent to $-x$ in the image coordinate frame. With this information, we can build our rotation matrix which translates from NWU to image coordinate space. Currently, the rotation matrix can only be provided as a `std::array<float3, 3>` to avoid confusion on storage order.
```cuda
std::array<float3, 3> rot;
rot[0] = make_float3( 0, -1, 0);
rot[1] = make_float3(-1,  0, 0);
rot[2] = make_float3( 0,  0, 1);
tex_helper_ptr->updateRotation(map_index, rot);
```

**Note:** We have been assuming that the map is aligned with the world frame and thus only the axes needed to be swapped/inverted. If the map is rotated with respect to the world frame, we will need to do a bit more work.
First, we shall notate the rotation from world frame to rotated map frame as $R_W^{map}$ .
We want to combine it with the coordinate transform rotation $R_{coord}$ which we do in the following order:

$$
\hat{R} = R_{coord} R^{map}_W
$$

However, putting $\hat{R}$ into \eqref{eq:coord_transform} would cause issues as our original origin is in map frame (we defined it by looking at where the top right corner was on the map). Thus, we need to update the origin to be in world frame to negate the effects of $R_W^{map}$:

$$
\begin{align}
\begin{bmatrix}
x_{W}^o \\
y_W^o \\
z_{W}^o
\end{bmatrix} &= \PP{R^{map}_W}^\top\begin{bmatrix}
x_{map}^o \\
y_{map}^o \\
z_{map}^o
\end{bmatrix} \\
\begin{bmatrix}
x^{image} \\
y^{image} \\
z^{image} \\
\end{bmatrix} &= R_{coord}R_W^{map} \PP{
\begin{bmatrix}
x^{robot}_W \\
y^{robot}_W \\
z^{robot}_W
\end{bmatrix}
- \begin{bmatrix}
x_{W}^o \\
y_W^o \\
z_{W}^o
\end{bmatrix}}
\end{align}
$$

The origin and rotation matrix to be passed to the texture helper would thus be $[x_{W}^o, y_W^o, z_{W}^o]^\top$ and $\hat{R}$ respectively.
### Map data
The final step is to actually provide the texture helper with the actual map data and then turn the texture on for querying purposes.
```cuda
bool column_major = false; // map_data was assumed row-major earlier
tex_helper_ptr->updateTexture(map_index, map_data, column_major);
tex_helper-ptr->enableTexture(map_index);
```
An `Eigen::MatrixXf` can also be used to pass in the map data. However, as Eigen uses column-order by default, the `updateTexture()` call might take longer as it converts the map to row-major order in preparation of use on the GPU. From here, all of the relevant map information will be passed to the GPU automatically when `cost.GPUSetup()` is called or if you have updated the map after allocating the GPU memory, calling `cost.paramsToDevice()` will update the GPU version of the map. When using the cost function with one of our MPPI controllers, `cost.GPUSetup()` is called in the Controller's constructor already so no explicit call is needed.

## Conclusion
This should provide all the necessary information for someone to get started using the Texture Helper API to
incorporate maps into your own Cost Function.
{% comment %}
You can even template on `float2` or `float4` (or the other datatype equivalents) in cases where you have
multiple data points that all need to be queried at the same point;
an example of such a datatype would be elevation and surface normals vectors of a terrain map.
Utilizing the `float2` or `float4` datatypes is more efficient/faster than using multiple `float` maps on the GPU
as CUDA can read up to 128 bytes (4 floats) in a single instruction, cutting down on the total number of memory reads.
{% endcomment %}
