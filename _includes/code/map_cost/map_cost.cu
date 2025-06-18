template <class CLASS_T, class DYN_T, class PARAMS_T>
MapCostImpl<CLASS_T, DYN_T, PARAMS_T>::MapCostImpl(cudaStream_t stream)
{
  tex_helper_ = new TwoDTextureHelper<float>(1, stream);
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
MapCostImpl<CLASS_T, DYN_T, PARAMS_T>::~MapCostImpl()
{
  delete tex_helper_;
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
void MapCostImpl<CLASS_T, DYN_T, PARAMS_T>::GPUSetup()
{
  PARENT_CLASS::GPUSetup();

  tex_helper_->GPUSetup();
  HANDLE_ERROR(cudaMemcpyAsync(&(this->cost_d_->tex_helper_), &(tex_helper_->ptr_d_), sizeof(TwoDTextureHelper<float>*),
                               cudaMemcpyHostToDevice, this->stream_));
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
void MapCostImpl<CLASS_T, DYN_T, PARAMS_T>::freeCudaMem()
{
  if (this->GPUMemStatus_)
  {
    tex_helper_->freeCudaMem();
  }
  PARENT_CLASS::freeCudaMem();
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
void MapCostImpl<CLASS_T, DYN_T, PARAMS_T>::paramsToDevice()
{
  if (this->GPUMemStatus_)
  {
    tex_helper_->copyToDevice();
  }
  PARENT_CLASS::paramsToDevice();
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
void MapCostImpl<CLASS_T, DYN_T, PARAMS_T>::bindToStream(cudaStream_t stream)
{
  PARENT_CLASS::bindToStream(stream);
  tex_helper_->bindToStream(stream);
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
float MapCostImpl<CLASS_T, DYN_T, PARAMS_T>::computeStateCost(const Eigen::Ref<const output_array> s, int timestep,
                                                              int* crash_status)
{
  float cost = 0.0;
  int map_index = 0;
  if (this->tex_helper_->checkTextureUse(map_index))
  {
    float3 query_point = make_float3(s(O_IND_CLASS(DYN_P, POS_X)), s(O_IND_CLASS(DYN_P, POS_Y)), 0.0);
    float distance = tex_helper_->queryTextureAtWorldPose(map_index, query_point);
    cost = fmaxf(2 - distance, 0.0);
  }
  return cost;
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
__device__ float MapCostImpl<CLASS_T, DYN_T, PARAMS_T>::computeStateCost(const float* __restrict__ s, int timestep,
                                                                         float* __restrict__ theta_c,
                                                                         int* __restrict__ crash_status)
{
  float cost = 0.0f;
  int map_index = 0;
  if (this->tex_helper_->checkTextureUse(map_index))
  {
    float3 query_point = make_float3(s[O_IND_CLASS(DYN_P, POS_X)], s[O_IND_CLASS(DYN_P, POS_Y)], 0.0);
    float distance = tex_helper_->queryTextureAtWorldPose(map_index, query_point);
    cost = fmaxf(2 - distance, 0.0);
  }
  return cost;
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
float MapCostImpl<CLASS_T, DYN_T, PARAMS_T>::terminalCost(const Eigen::Ref<const output_array> s)
{
  return 0.0;
}

template <class CLASS_T, class DYN_T, class PARAMS_T>
__device__ float MapCostImpl<CLASS_T, DYN_T, PARAMS_T>::terminalCost(const float* __restrict__ s,
                                                                     float* __restrict__ theta_c)
{
  return 0.0;
}
