import { http } from "msw";
import { HttpResponse } from "msw";
import { customRender } from "@/tests";
import server from "@/tests/mocks/node";
import SystemBackupsView from ".";

describe("System Backups", () => {
  it("should render with backups", async () => {
    server.use(
      http.get("/api/system/backups", () => {
        return HttpResponse.json({
          data: [],
        });
      }),
    );

    customRender(<SystemBackupsView />);

    // TODO: Assert
  });
});
