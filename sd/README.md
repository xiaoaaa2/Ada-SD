# Improved AISHELL4 Speaker Diarization


## Usage
In speaker diarization process, we will only use the first channel as the SAD model and the Speaker-Embedding-Extrator do not support the multi-channel wav. We changed the `./VBx/predict.py` to use the msr network and fine-tuned speaker embedding extractor.

> You can find all the process in `sd/run.sh` and the comments in it. You can experiment with any segment length and segment jump within the specified range, simply by modifying the parameters in the file.



**Here is the main stage:**
1. You should first prepare the key file `wav.scp` for our tools, the scripts `local/make_aishell4_test.sh` is just a sample, you can use any language (like perl, python or shell) as you like. **If you find some naming problems when use, you can replace this step by your own script to prepare the files**.
2. You should change your own data path and `kaldi-root` first in `run.sh` and `path.sh`.  The script `local/do_segmentations.sh` is to get the SAD result for the future work. You will find the segments file under the `$sad_result_dir`.
3. When use the `VBx` tools for the diarization, you should convert the segments to the `.lab`. Use `scripts/segment_to_lab.sh` to change the file format
4. The speaker diarization code needs two stage the speaker-embedding extract and the speaker-embedding cluster. Our baseline use the `VBx` tools to extract speaker-embeddings. The feature-extractor is inside, you don't have to prepare the feature before. **Note our scripts run in a SGE systems sor we sub the extract-embedding jobs to the queue.pl, if you do not have it, try to extract the speaker embeddings one by one**. Besides, we recommand you add a `exit 1` after the stage 3 to waiting for the extracting process finished.
5. For the speaker-embedding cluster, you can use the `run_cluster.sh` and the code will make the rttm for each audio in the wav.scp.



## Model Download

You need download the model from the [path](https://data-tx.oss-cn-hangzhou.aliyuncs.com/AISHELL-4-Code/sd-part.zip), you should mv the `exp` to the `sd/` and the `ResNet101_16kHz` to the `VBx/models`.




## Reference
1. [kaldi-sad-model](http://kaldi-asr.org/models/m12)
2. [VBx](https://github.com/BUTSpeechFIT/VBx)

