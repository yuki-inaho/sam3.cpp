// Four small pointer projections were originally CPU float loops. Route their
// opt-in Q8 variant through the same GGML dispatch as every other Linear,
// including dynamic activation quantization; never silently dequantize only W.
static void sam2_q8_linear_vector(const sam3_model& model, ggml_tensor* weight,
                                 ggml_tensor* bias, const float* input, float* output) {
    const ggml_init_params params={ggml_tensor_overhead()*32+ggml_graph_overhead(),nullptr,true};
    auto* ctx=ggml_init(params);
    if(!ctx) throw std::runtime_error("Q8 pointer Linear context allocation failed");
    auto* x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,weight->ne[0],1);
    ggml_set_input(x);
    auto* y=ggml_add(ctx,ggml_mul_mat(ctx,weight,x),bias);
    ggml_set_output(y);
    auto* graph=ggml_new_graph_custom(ctx,32,false);
    ggml_build_forward_expand(graph,y);
    auto* alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    bool ok=ggml_gallocr_alloc_graph(alloc,graph);
    if(ok) {
        ggml_backend_tensor_set(x,input,0,weight->ne[0]*sizeof(float));
        ok=sam3_graph_compute(model.backend,graph,1);
    }
    if(ok) ggml_backend_tensor_get(y,output,0,weight->ne[1]*sizeof(float));
    ggml_gallocr_free(alloc); ggml_free(ctx);
    if(!ok) throw std::runtime_error("Q8 pointer Linear graph failed");
}
