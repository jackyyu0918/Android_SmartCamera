package com.jackyyu0918.finalyearproject_54820425;

import android.media.MediaRecorder;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class RecordingHandler {
    private MediaRecorder recorder;
    private static final RecordingHandler RecordingHandler= new RecordingHandler();

    private RecordingHandler(){
        this.recorder = new MediaRecorder();
    }

    public static RecordingHandler getInstance(){
        return RecordingHandler;
    }

    //Prepare
    public void prepareRecorder(Zoomcameraview mOpenCvCameraView) {
        //Success for start,but shut down on stop

        //Video source is from the surface
        //Everything draw on the surface will be recorder by recorder
        recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);

        //Store the video with time stamp
        String currentDateandTime = generateDateInfo();

        recorder.setOutputFile(Environment.getExternalStorageDirectory().getAbsoluteFile() + File.separator + "/FYP/" + currentDateandTime + ".mp4");
        recorder.setVideoEncodingBitRate(1000000);
        recorder.setVideoFrameRate(60);
        recorder.setVideoSize(mOpenCvCameraView.getWidth(), mOpenCvCameraView.getHeight());
        recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        recorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);

        try {
            recorder.prepare();
            System.out.println("success prepared media recorder!!");

        } catch (IllegalStateException e) {
            Log.e("debug mediarecorder", "not prepare");
        } catch (IOException e) {
            Log.e("debug mediarecorder", "not prepare IOException");
        }

        //Initialized the mRecorder in CameraBridgeViewBase and make a new surface for recording!
        //Everything draw on that
        mOpenCvCameraView.setRecorder(recorder);

        //Only Audio
        //Success!
            /*
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
            recorder.setOutputFile(Environment.getExternalStorageDirectory().getAbsoluteFile()+File.separator+"outputAudio.3gp");
             */
    }

    public void startRecord(boolean isRecording) {
        recorder.start();
        isRecording = true;
    }

    public void stopRecord() {
        try {
            recorder.stop();
        } catch (RuntimeException stopException) {
            System.out.println("RuntimeException occurred!");
        }
    }

    public void resetRecorder(){
        recorder.reset();
    }


    public String generateDateInfo() {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault());
        String currentDateandTime = sdf.format(new Date());
        return currentDateandTime;
    }

    public MediaRecorder getMediaRecorder(){
        return recorder;
    }



}
