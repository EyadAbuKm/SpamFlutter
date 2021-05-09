import 'package:flutter/services.dart';

// Import tflite_flutter
import 'package:tflite_flutter/tflite_flutter.dart';

class SpamClassifier {
  // name of the model file
  final _modelFile = 'lite-model_tutorials_spam-detection_tflite_1.tflite';
  final _vocabFile = 'count_dict.txt';

  // Maximum length of sentence
  final int _sentenceLen = 10000;

  final String start = '<START>';
  final String pad = '<PAD>';
  final String unk = '<UNKNOWN>';

  late Map<String, int> _dict;
  var reversed;

  // TensorFlow Lite Interpreter object
  late Interpreter _interpreter;

  SpamClassifier() {
    // Load model when the classifier is initialized.
    _loadModel();
    _loadDictionary();
  }

  void _loadModel() async {
    // Creating the interpreter using Interpreter.fromAsset
    _interpreter = await Interpreter.fromAsset(_modelFile);
    print('Interpreter loaded successfully');
  }

  void _loadDictionary() async {
    final vocab = await rootBundle.loadString('assets/$_vocabFile');
    var dict = <String, int>{};
    final vocabList = vocab.split('\n');
    for (var i = 0; i < 7678; i++) {
      var entry = vocabList[i].trim().split(' ');
      // print(entry);
      dict[entry[0]] = int.parse(entry[1]);
    }
    reversed = dict.map((k, v) => MapEntry(v, k));

    print(reversed[140]);
    print(dict['on']);
    print(dict.containsKey('onsafsaf'));
    // print(dict[1]);

    // print(dict);
    _dict = dict;
    print('Dictionary loaded successfully');
  }

  List<double> classify(String rawText) {
    // tokenizeInputText returns List<List<double>>
    // of shape [1, 256].
    List<double> input = tokenizeInputText(rawText);

    // output of shape [1,2].
    var output = List<double>.filled(2, 0).reshape([1, 2]);

    // The run method will run inference and
    // store the resulting values in output.
    _interpreter.run(input, output);

    return [output[0][0], output[0][1]];
  }

  List<double> tokenizeInputText(String text) {
    // Whitespace tokenization
    final toks = text.split(' ');
    print('hello from toks');
    print(toks);

    // Create a list of length==_sentenceLen filled with the value <pad>
    var vec = List<double>.filled(_sentenceLen, 0);
    // print(vec[0]);
    var index = 0;
    // if (_dict.containsKey(start)) {
    //   vec[index++] = _dict[start]!.toDouble();
    // }

    // For each word in sentence find corresponding index in dict
    for (var tok in toks) {
      if (index > _sentenceLen) {
        break;
      }
      // if (tok == "go") {
      //   print('hello from go');
      //   print(_dict[tok]);
      // }
      if (_dict.containsKey(tok)) {
        vec[_dict[tok]!] = 1;
        // print(vec[_dict[tok]!]);
      }
    }
    print(vec[0]);
    print(vec[140]);
    print(vec[2]);
    // returning List<List<double>> as our interpreter input tensor expects the shape, [1,256]
    return vec;
  }
}
