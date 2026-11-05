/* Independent review records; no automatic aesthetic verdict. */
(function (root) {
  'use strict';
  function buildSubmission(caseIds, choices) {
    const valid = new Set(['A', 'B', 'equal', 'neither']);
    if (!Array.isArray(caseIds) || new Set(caseIds).size !== caseIds.length ||
        caseIds.some(id => typeof id !== 'string' || !id)) throw new Error('Invalid case IDs');
    const ratings = caseIds.map(id => {
      const choice = choices[id] === undefined ? null : choices[id];
      if (choice !== null && !valid.has(choice)) throw new Error('Invalid review choice');
      return {case_id: id, choice};
    });
    return {schema_version: 1, study: 'visible-face-v4-natural-pairs',
      completed: ratings.every(row => row.choice !== null), ratings,
      limitation: 'Subjective preview review; no objective style ground truth'};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {buildSubmission};
  else root.buildSubmission = buildSubmission;
})(typeof globalThis !== 'undefined' ? globalThis : this);
