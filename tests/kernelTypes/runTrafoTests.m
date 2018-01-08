% runs all tranformation tests and displays summary

import matlab.unittest.TestSuite
[testDir,~] = fileparts(which('runTrafoTests'));
suiteFolder = TestSuite.fromFolder(testDir);
result = run(suiteFolder);