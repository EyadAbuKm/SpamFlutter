import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:sklite/SVM/SVM.dart';
import 'package:sklite/utils/io.dart';

import 'package:tflite_flutter_plugin_example/classifier.dart';
import 'package:tflite_flutter_plugin_example/dc.dart';
import 'package:tflite_flutter_plugin_example/spam_classifier.dart';

void main() => runApp(MyApp());

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late TextEditingController _controller;
  late SpamClassifier _classifier;
  late List<Widget> _children;
  late SVC svc;
  @override
  void initState() {
    super.initState();
    loadModel("assets/svc_count.json").then((x) {
      this.svc = SVC.fromMap(json.decode(x));
    });
    _controller = TextEditingController();
    _classifier = SpamClassifier();
    _children = [];
    _children.add(Container());
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.orangeAccent,
          title: const Text('Text classification'),
        ),
        body: Container(
          padding: const EdgeInsets.all(4),
          child: Column(
            children: <Widget>[
              Expanded(
                  child: ListView.builder(
                itemCount: _children.length,
                itemBuilder: (_, index) {
                  return _children[index];
                },
              )),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                    border: Border.all(color: Colors.orangeAccent)),
                child: Row(children: <Widget>[
                  Expanded(
                    child: TextField(
                      decoration: const InputDecoration(
                          hintText: 'Write some text here'),
                      controller: _controller,
                    ),
                  ),
                  TextButton(
                    child: const Text('Classify'),
                    onPressed: () {
                      final text = _controller.text;
                      // final prediction = _classifier.tokenizeInputText(text);
                      final prediction = DecisionTree.score(
                          _classifier.tokenizeInputText(text));
                      // svc.predict((_classifier.tokenizeInputText(text)));
                      // print(prediction[0]);
                      setState(() {
                        _children.add(Dismissible(
                          key: GlobalKey(),
                          onDismissed: (direction) {},
                          child: Card(
                            child: Container(
                              padding: const EdgeInsets.all(16),
                              color: prediction[0] > 0.5
                                  ? Colors.lightGreen
                                  : Colors.redAccent,
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: <Widget>[
                                  Text(
                                    "Input: $text",
                                    style: const TextStyle(fontSize: 16),
                                  ),
                                  Text("Output:"),
                                  Text("   Spam: ${prediction[0]}"),
                                  Text("   Ham: ${1 - prediction[0]}"),
                                ],
                              ),
                            ),
                          ),
                        ));
                        _controller.clear();
                      });
                    },
                  ),
                ]),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
// flutter run --no-sound-null-safety
