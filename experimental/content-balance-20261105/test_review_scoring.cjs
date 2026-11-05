'use strict';
const assert = require('node:assert/strict');
const {buildSubmission} = require('./review_scoring.js');
assert.deepEqual(buildSubmission(['case1', 'case2'], {case1: 'A'}).ratings,
  [{case_id: 'case1', choice: 'A'}, {case_id: 'case2', choice: null}]);
assert.equal(buildSubmission(['case1', 'case2'], {case1: 'A'}).completed, false);
assert.equal(buildSubmission(['case1'], {case1: 'neither'}).completed, true);
assert.equal(buildSubmission(['case1'], {case1: 'equal'}).ratings[0].choice, 'equal');
assert.throws(() => buildSubmission(['case1'], {case1: 'approve'}));
assert.throws(() => buildSubmission(['case1', 'case1'], {}));
console.log('Review record validation passed: unanswered, ties, neither, invalid choices and duplicate IDs');
