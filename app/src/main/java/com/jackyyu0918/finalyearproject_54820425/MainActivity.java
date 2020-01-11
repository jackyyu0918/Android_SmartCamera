package com.jackyyu0918.finalyearproject_54820425;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerBoosting;
import org.opencv.tracking.TrackerCSRT;
import org.opencv.tracking.TrackerGOTURN;
import org.opencv.tracking.TrackerKCF;
import org.opencv.tracking.TrackerMIL;
import org.opencv.tracking.TrackerMOSSE;
import org.opencv.tracking.TrackerMedianFlow;
import org.opencv.tracking.TrackerTLD;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.util.List;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean              mIsJavaCamera = true;
    private MenuItem             mItemSwitchCamera = null;
    private static final int CAMERA_REQUEST = 1888;

    //My code
    //Matrix
    private Mat mRgba;
    private Mat mGray;
    private Mat testMat;
    private Mat videoMat;

    //Tracker
    //private TrackerKCF firstTracker;
    private Tracker objectTracker;
    private boolean isInitTracker = false;

    private Rect2d roi;
    private Point trackerCoordinate;
    private Point trackerSize;
    private Scalar greenColorOutline;

    private Rect roiRect;

    //button for init tracker
    private Button button_startTrack;
    private String currentTrackerType;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = findViewById(R.id.main_surfaceView);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, CAMERA_REQUEST);
        }


        //Tracker section
        //roi setting
        trackerCoordinate = new Point(700,200);
        trackerSize = new Point(300,300);
        greenColorOutline = new Scalar(0, 255, 0, 255);

        roi = new Rect2d(trackerCoordinate.x,trackerCoordinate.y,trackerSize.x,trackerSize.y);
        roiRect = new Rect((int)trackerCoordinate.x,(int)trackerCoordinate.y,(int)trackerSize.x,(int)trackerSize.y);

        //tracker creation, base on drop down menu selection
        //currentTrackerType = "KCF";

        //spinner tracker selection
        Spinner trackerSpinner  = findViewById(R.id.trackerSpinner);

        final ArrayAdapter<String> nameAdaptar = new ArrayAdapter<String>(MainActivity.this,android.R.layout.simple_expandable_list_item_1, getResources().getStringArray(R.array.trackingAlgorithmName));

        nameAdaptar.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        trackerSpinner.setAdapter(nameAdaptar);


        trackerSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                Toast.makeText(MainActivity.this, "You are choosing "+ parent.getSelectedItem().toString() + ".", Toast.LENGTH_SHORT).show() ;
                currentTrackerType = parent.getSelectedItem().toString();
                System.out.println("currentTrackerType: " + currentTrackerType);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                Toast.makeText(MainActivity.this, "Nothing is selected.", Toast.LENGTH_LONG).show();
            }
        });

        //button onClick listener
        //start button
        Button button_startTrack = findViewById(R.id.button_startTrack);
        button_startTrack.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view)
            {
                //No default tracker type
                //firstTracker = TrackerKCF.create();

                //own function, create proper tracker
                createTracker(currentTrackerType);

                //tracker initialization
                objectTracker.init(mGray, roi);
                //System.out.println("Tracker init result: " + firstTracker.init(mGray,roi));
                isInitTracker = true;

                //Tracker message
                Toast toast = Toast.makeText(MainActivity.this,
                        "Current tracker: " + objectTracker.getClass(), Toast.LENGTH_LONG);
                //顯示Toast
                toast.show();
            }
        });
        //reset button
        Button button_resetTrack = findViewById(R.id.button_resetTracker);
        button_resetTrack.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view)
            {
                objectTracker = null;
                isInitTracker = false;
            }
        });
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
        testMat = Mat.zeros(100,100,CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        Size sizeRgba = mRgba.size();
        Mat rgbaInnerWindow;
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;

        //tracking section
        // if initialized tracker, start update the ROI
        if(isInitTracker){
            roiRect.x = (int)roi.x;
            roiRect.y = (int)roi.y;
            roiRect.width = (int)roi.width;
            roiRect.height = (int)roi.height;

            System.out.println("Tracker update result: " + objectTracker.update(mGray,roi));

            //top-left corner
            Mat zoomCorner = mRgba.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);

            //matrix at the middle
            //Mat mZoomWindow = mRgba.submat(rows / 2 - 9 * rows / 100, rows / 2 + 9 * rows / 100, cols / 2 - 9 * cols / 100, cols / 2 + 9 * cols / 100);
            Mat mZoomWindow = mRgba.submat((int)(roi.y), (int)(roi.y+roi.height), (int)(roi.x), (int)(roi.x+roi.width));

            //show middle matrix at the top-left corner
            Imgproc.resize(mZoomWindow, zoomCorner, zoomCorner.size(), 0, 0, Imgproc.INTER_LINEAR_EXACT);
            //Imgproc.resize(mZoomWindow, zoomCorner, zoomCorner.size(), 0, 0, Imgproc.INTER_LINEAR_EXACT);
            Size wwsize = mZoomWindow.size();
            Imgproc.rectangle(mZoomWindow, new Point(1, 1), new Point(wwsize.width - 2, wwsize.height - 2), new Scalar(255, 0, 0, 255), 2);
            zoomCorner.release();
            mZoomWindow.release();
        }

        //draw rectangle using Rect roiRect
        Imgproc.rectangle(mRgba,roiRect,greenColorOutline,2);

        return mRgba;

    }

    //my own function
    public void createTracker(String trackerType){
        switch (trackerType){
            case "KCF":
                System.out.println("KCF case.");
                objectTracker = TrackerKCF.create();
                break;
            case "MedianFlow":
                System.out.println("MedianFlow case.");
                objectTracker = TrackerMedianFlow.create();
                break;
            case "TLD":
                System.out.println("TLD case.");
                objectTracker = TrackerTLD.create();
                break;
            case "Boosting":
                System.out.println("Boosting case.");
                objectTracker = TrackerBoosting.create();
                break;
            case "CSRT":
                System.out.println("CSRT case.");
                objectTracker = TrackerCSRT.create();
                break;
            case "GOTURN":
                System.out.println("GOTURN case.");
                objectTracker = TrackerGOTURN.create();
                break;
            case "MIL":
                System.out.println("MIL case.");
                objectTracker = TrackerMIL.create();
                break;
            case "MOSSE":
                System.out.println("MOSSE case.");
                objectTracker = TrackerMOSSE.create();
                break;
            default:
                break;
        }
    }
}