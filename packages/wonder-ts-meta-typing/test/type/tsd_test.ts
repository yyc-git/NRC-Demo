import tsd from 'tsd';

describe("test tsd", () => {
	beforeEach(() => {
	});

	test("test types", () => {
		return tsd({
			cwd: process.cwd(),
			// typingsFile: "./src/**/index.d.ts",
			testFiles: ["./src/**/index.test-d.ts"]
		}).then((diagnostics) => {
			let errorMessage = [];

			diagnostics.forEach(test => {
				errorMessage.push(`${test.fileName}:${test.line}:${test.column} - ${test.severity} - ${test.message}`);
			});

			// expect(diagnostics.length).toEqual(0);
			expect(errorMessage).toEqual([]);
		})

	})
})