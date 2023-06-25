import 'package:flutter/material.dart';
import 'package:yoga_app/screens/pose_page.dart';
// import 'package:flutter/src/widgets/framework.dart';
// import 'package:flutter/src/widgets/placeholder.dart';

class YogaPose {
  final String image;
  final String name;
  YogaPose(this.image, this.name);
}

class HomeBody extends StatefulWidget {
  const HomeBody({super.key});

  @override
  State<HomeBody> createState() => _HomeBodyState();
}

class _HomeBodyState extends State<HomeBody> {
  final List<YogaPose> _items=[
    YogaPose("https://images.unsplash.com/photo-1545389336-cf090694435e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTV8fHlvZ2F8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&w=1400&q=60", "Tree Pose"),
    YogaPose("https://images.unsplash.com/photo-1545389336-cf090694435e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTV8fHlvZ2F8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&w=1400&q=60", "Pose 2"),
    YogaPose("https://images.unsplash.com/photo-1545389336-cf090694435e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTV8fHlvZ2F8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&w=1400&q=60", "Pose 3"),
    YogaPose("https://images.unsplash.com/photo-1545389336-cf090694435e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTV8fHlvZ2F8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&w=1400&q=60", "Pose 4"),
    YogaPose("https://images.unsplash.com/photo-1545389336-cf090694435e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTV8fHlvZ2F8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&w=1400&q=60", "Pose 5"),
  ];

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: SingleChildScrollView(
        scrollDirection: Axis.vertical,
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(
                children: [
                  Container(
                    width: 300,
                    height: 50,
                    padding: const EdgeInsets.fromLTRB(20, 0, 0, 0),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.9),
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
                      alignment: Alignment.centerLeft,
                      child: Text(
                        "All Yoga Poses",
                        style: TextStyle(
                          fontSize: 23,
                          fontFamily: 'PlusJakartaSans',
                          fontWeight: FontWeight.bold,
                        ),
                      )
                    ),

                  ),
                  const SizedBox(
                    width: 20,
                    height: 50,
                  ),
                  Container(
                    width: 50,
                    height: 50,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.9),
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
                    child: const Icon(Icons.search_rounded),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(
                children: [
                  Container(
                    width: 50,
                    height: 32,
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
                          "All",
                          style: TextStyle(
                            fontSize: 13,
                            fontFamily: 'PlusJakartaSans',
                            fontWeight: FontWeight.bold,
                          ),
                        )
                    ),

                  ),
                  const SizedBox(
                    width: 15,
                    height: 50,
                  ),
                  Container(
                    width: 87,
                    height: 32,
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
                          "Sitting",
                          style: TextStyle(
                            fontSize: 13,
                            fontFamily: 'PlusJakartaSans',
                            fontWeight: FontWeight.bold,
                          ),
                        )
                    ),

                  ),
                  const SizedBox(
                    width: 15,
                    height: 50,
                  ),
                  Container(
                    width: 87,
                    height: 32,
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
                          "Intense",
                          style: TextStyle(
                            fontSize: 13,
                            fontFamily: 'PlusJakartaSans',
                            fontWeight: FontWeight.bold,
                          ),
                        )
                    ),

                  ),
                  const SizedBox(
                    width: 15,
                    height: 50,
                  ),
                  Container(
                    width: 87,
                    height: 32,
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
                          "Standing",
                          style: TextStyle(
                            fontSize: 13,
                            fontFamily: 'PlusJakartaSans',
                            fontWeight: FontWeight.bold,
                          ),
                        )
                    ),

                  ),
                  const SizedBox(
                    width: 15,
                    height: 50,
                  ),
                ],
              ),
            ),
            Padding(
                padding: const EdgeInsets.symmetric(horizontal: 10),
                child: Container(
                  padding: const EdgeInsets.all(10),
                  height: 1500,
                  child: GridView.builder(
                      itemCount: _items.length,
                      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 2,
                        crossAxisSpacing: 4,
                        mainAxisSpacing: 4,
                        mainAxisExtent: 255,
                      ),
                      // scrollDirection: Axis.vertical,
                      physics: const ScrollPhysics(),
                      itemBuilder: (context,index){
                        return _buildChild(index);
                      }),
                ),
            ),

          ]
        ),
      )
    );
  }

  Widget _buildChild(int index){
    return InkWell(
      // splashColor:
      // Constants.lightBlue,
      // highlightColor:
      // Constants.transparent,
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => const PosePage(),
          ),
        );
      },
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.9),
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
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              height: 220,
              child: Center(
                child: Image.network(_items[index].image),
              ),
            ),
            Text(
              _items[index].name,
              style: const TextStyle(
                fontSize: 18,
                fontFamily: 'PlusJakartaSans',
                fontWeight: FontWeight.w700,
              ),
            )
          ],
        ),
      ),
    );
  }
}