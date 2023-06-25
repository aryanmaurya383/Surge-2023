import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:yoga_app/main.dart';
import 'package:yoga_app/api.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({Key? key}) : super(key: key);

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraImage? cameraImage;
  CameraController? cameraController;

  @override
  void initState(){
    super.initState();
    loadCamera();
  }

  loadCamera(){
    cameraController=CameraController(cameras![0], ResolutionPreset.medium);
    cameraController!.initialize().then((value){
      if(!mounted){
        return;
      }
      else{
        setState(() {
          cameraController!.startImageStream((imageStream) {
            cameraImage=imageStream;
            runModel();
          });
        });
      }
    });
  }

  runModel()async{
    if(cameraImage!=null){

    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Padding(padding: EdgeInsets.all(20),
            child: Container(
              height: MediaQuery.of(context).size.height*0.7,
              width: MediaQuery.of(context).size.width,
              child: !cameraController!.value.isInitialized?
                  Container():
                  AspectRatio(aspectRatio: cameraController!.value.aspectRatio,
                  child: CameraPreview(cameraController!),)
            ),
          ),
          Text(
            "Camera opened",
            style: TextStyle(
            fontSize: 20,
            fontFamily: 'PlusJakartaSans',
            fontWeight: FontWeight.bold,
            )
          )
        ],
      ),
    );
  }
}
