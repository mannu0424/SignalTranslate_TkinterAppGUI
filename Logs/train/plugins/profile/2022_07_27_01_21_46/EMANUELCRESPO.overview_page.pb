?$	??????S@?ɧ/>?[@ގpZ????!l#?	?c@	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsl#?	?c@1?#?]Ja4@I??d??5a@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsގpZ????1??C6?.??I???6???*	43333Sc@2U
Iterator::Model::ParallelMapV2_?Qګ?!??@??A@)_?Qګ?1??@??A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?I+???!
?+?u<@)e?X???1[? ?]a6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??A?f??!?樚	+@)?:pΈ??1?c??Nj'@:Preprocessing2F
Iterator::Model o?ŏ??!l?x??/F@)?ZӼ???1U3?x?^"@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?46<??!?Q5?wB@)???<,Ԋ?1"??k? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceU???N@??!?*??!R@)U???N@??1?*??!R@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipj?t???!?s?{O?K@)????Mb??1???ղ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???f?!3?x?^???)Ǻ???f?13?x?^???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?87.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??(??U@Qz???ї)@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!       	!       "$	????f$@`gC???,@??C6?.??!?#?]Ja4@*	!       2	!       :$	????`Q@?\???X@???6???!??d??5a@B	!       J	!       R	!       Z	!       b	!       JGPUb q??(??U@yz???ї)@?"?
?gradient_tape/sequential/lstm_2/while/sequential/lstm_2/while_grad/body/_1778/gradient_tape/sequential/lstm_2/while/gradients/AddN_6AddN?,????!?,????"?
?gradient_tape/sequential/lstm_1/while/sequential/lstm_1/while_grad/body/_2143/gradient_tape/sequential/lstm_1/while/gradients/AddN_6AddNs?h????!2?<͔?"f
Ksequential/lstm_2/while/body/_707/sequential/lstm_2/while/lstm_cell_2/splitSplit???q?V??!,c?h????"?
?gradient_tape/sequential/lstm_2/while/sequential/lstm_2/while_grad/body/_1778/gradient_tape/sequential/lstm_2/while/gradients/sequential/lstm_2/while/lstm_cell_2/split_grad/concatConcatV2???8݃}?!?&?۽???"?
?gradient_tape/sequential/lstm/while/sequential/lstm/while_grad/body/_2508/gradient_tape/sequential/lstm/while/gradients/sequential/lstm/while/lstm_cell/split_grad/concatConcatV2?~???ly?!??Vڥ?"f
Ksequential/lstm_1/while/body/_354/sequential/lstm_1/while/lstm_cell_1/splitSplit?z?x?!???????"^
Csequential/lstm/while/body/_1/sequential/lstm/while/lstm_cell/splitSplitv?? 
?x?!o??Sg??"g
Lsequential/lstm_3/while/body/_1060/sequential/lstm_3/while/lstm_cell_3/splitSplit?ɗ?rx?!ͺӦ???"?
?gradient_tape/sequential/lstm_1/while/sequential/lstm_1/while_grad/body/_2143/gradient_tape/sequential/lstm_1/while/gradients/sequential/lstm_1/while/lstm_cell_1/split_grad/concatConcatV2be(??w?!?-?????"?
?gradient_tape/sequential/lstm_3/while/sequential/lstm_3/while_grad/body/_1413/gradient_tape/sequential/lstm_3/while/gradients/sequential/lstm_3/while/lstm_cell_3/split_grad/concatConcatV2?y?8?v?!???oIt??Q      Y@Y?"?a???a?7O???X@q?d7b?uW@y???`#b{?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?87.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?93.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 