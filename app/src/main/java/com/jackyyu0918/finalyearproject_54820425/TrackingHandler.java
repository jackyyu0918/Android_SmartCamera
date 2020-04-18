package com.jackyyu0918.finalyearproject_54820425;

import android.widget.Toast;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerKCF;
import org.opencv.tracking.TrackerMOSSE;

//Singleton class
public class TrackingHandler {
    private Tracker objectTracker;
    private Rect2d roiRect2d;
    private Rect roiRect;
    private Rect targetObjectRect;

    //Pre-defined size
    private Point trackerCoordinate;
    private Point trackerSize;
    private Scalar greenColorOutline;

    private static final TrackingHandler TrackingHandler= new TrackingHandler();
    private TrackingHandler(){
        this.objectTracker = null;
        this.greenColorOutline = new Scalar(0, 255, 0, 255);
        this.trackerCoordinate = new Point();
        this.trackerSize = new Point();

        this.roiRect2d = new Rect2d();
        this.roiRect = new Rect();
        //tracking rect
        this.targetObjectRect = new Rect();
    }

    public static TrackingHandler getInstance(){
        return TrackingHandler;
    }


    //Create Tracker based on the passed string
    public void createTracker(String trackerType) {
        switch (trackerType) {
            case "KCF":
                System.out.println("KCF case.");
                objectTracker = TrackerKCF.create();
                break;
            case "MOSSE":
                System.out.println("MOSSE case.");
                objectTracker = TrackerMOSSE.create();
                break;
            default:
                break;
        }
    }



    public Tracker getTracker(){
        return TrackingHandler.objectTracker;
    }

    public void setTracker(Tracker tracker){
        objectTracker = tracker;
    }

    public boolean isInitTracker(){
        return objectTracker != null;
    }

    public void setTrackerSize(int x, int y, int width, int height) {
        roiRect.x = x;
        roiRect.y = y;
        roiRect.width = width;
        roiRect.height = height;

        roiRect2d.x = x;
        roiRect2d.y = y;
        roiRect2d.width = width;
        roiRect2d.height = height;
    }

    public void resetTrackerDetails() {
        trackerCoordinate = new Point();
        trackerSize = new Point();

        roiRect.x = (int) trackerCoordinate.x;
        roiRect.y = (int) trackerCoordinate.y;
        roiRect.width = (int) trackerSize.x;
        roiRect.height = (int) trackerSize.y;

        roiRect2d.x = trackerCoordinate.x;
        roiRect2d.y = trackerCoordinate.y;
        roiRect2d.width = trackerSize.x;
        roiRect2d.height = trackerSize.y;
    }

    //Set rectangle info from drag class
    public void calculateRectInfo(android.graphics.Point[] points) {
        trackerCoordinate.x = points[0].x + 40;
        trackerCoordinate.y = points[0].y + 20;
        trackerSize.x = ((points[2].x + 40) - (points[0].x + 40));
        trackerSize.y = ((points[2].y + 20) - (points[0].y + 20));

        roiRect2d.x = trackerCoordinate.x;
        roiRect2d.y = trackerCoordinate.y;
        roiRect2d.width = trackerSize.x;
        roiRect2d.height = trackerSize.y;

        roiRect.x = (int) trackerCoordinate.x;
        roiRect.y = (int) trackerCoordinate.y;
        roiRect.width = (int) trackerSize.x;
        roiRect.height = (int) trackerSize.y;
    }



    public void sysnRectValue(){
        roiRect.x = (int) roiRect2d.x;
        roiRect.y = (int) roiRect2d.y;
        roiRect.width = (int) roiRect2d.width;
        roiRect.height = (int) roiRect2d.height;
    }

    //================Important function for tracking==================//
    public boolean initializeTracker(Mat mat){
        return objectTracker.init(mat, roiRect2d);
    }

    public boolean updateTracker(Mat mat){
        return objectTracker.update(mat, roiRect2d);
    }

    //Get method
    public Rect getroiRect(){
        return roiRect;
    }

    public Rect2d getRoiRect2d(){
        return roiRect2d;
    }

    public Rect gettargetObjectRect(){
        return targetObjectRect;
    }

    public Scalar getGreenColorOutline(){
        return greenColorOutline;
    }
}
