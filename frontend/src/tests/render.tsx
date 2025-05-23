import { FunctionComponent } from "react";
import { customRender } from ".";

export interface RenderTestCase {
  name: string;
  ui: FunctionComponent;
  setupEach?: () => void;
}

export function renderTest(name: string, cases: RenderTestCase[]) {
  describe(name, () => {
    beforeEach(() => {
      cases.forEach((element) => {
        element.setupEach?.();
      });
    });

    cases.forEach((element) => {
      it(`${element.name.toLowerCase()} should render`, () => {
        customRender(<element.ui />);
      });
    });
  });
}
