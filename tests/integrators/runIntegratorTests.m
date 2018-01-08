% executes all block tests and displays summary

import matlab.unittest.TestSuite
[testDir,~] = fileparts(which('runIntegratorTests'));
suiteFolder = TestSuite.fromFolder(testDir);
result = run(suiteFolder);