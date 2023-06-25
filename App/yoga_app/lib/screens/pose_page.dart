import 'package:flutter/material.dart';
import 'package:yoga_app/widgets/pose_page_body.dart';

class PosePage extends StatefulWidget {
  const PosePage({Key? key}) : super(key: key);

  @override
  State<PosePage> createState() => _PosePageState();
}

class _PosePageState extends State<PosePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: PoseBody(),
    );
  }
}
