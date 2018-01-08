% runs tests for all layers and displays summary

import matlab.unittest.TestSuite
[testDir,~] = fileparts(which('runLayerTests'));
suiteFolder = TestSuite.fromFolder(testDir);
result = run(suiteFolder);