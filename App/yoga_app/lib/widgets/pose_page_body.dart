import 'package:flutter/material.dart';
import 'package:yoga_app/screens/camera_page.dart';


class PoseBody extends StatefulWidget {
  const PoseBody({Key? key}) : super(key: key);

  @override
  State<PoseBody> createState() => _PoseBodyState();
}

class _PoseBodyState extends State<PoseBody> {

  @override
  Widget build(BuildContext context) {
    return SafeArea(
        child: SingleChildScrollView(
          scrollDirection: Axis.vertical,
          child: Column(
              children: [

                SizedBox(
                  height: 300,
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: Align(
                    alignment: Alignment.center,
                    child: InkWell(
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => const CameraPage(),
                          ),
                        );
                      },
                      child: Container(
                        width: 200,
                        height: 70,
                        // padding: const EdgeInsets.fromLTRB(0, 0, 0, 0),
                        decoration: BoxDecoration(
                          color: Colors.teal.withOpacity(0.5),
                          borderRadius: BorderRadius.circular(15),
                          boxShadow: const [
                            BoxShadow(
                              color: Color(0xffDDDDDD),
                              blurRadius: 6.0, // soften the shadow
                              spreadRadius: 1.0, //extend the shadow
                              offset: Offset(
                                0.0, // Move to right 5  horizontally
                                0.0, // Move to bottom 5 Vertically
                              ),
                            ),
                          ],
                        ),
                        child: const Align(
                            alignment: Alignment.center,
                            child: Text(
                              "Open Camera",
                              style: TextStyle(
                                fontSize: 20,
                                fontFamily: 'PlusJakartaSans',
                                fontWeight: FontWeight.bold,
                              ),
                            )
                        ),

                      ),
                    ),
                  ),
                ),

              ]
          ),
        )
    );
  }
}
