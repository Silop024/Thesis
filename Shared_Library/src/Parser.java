import java.util.Map;

public class Parser
{
    public Map<Construct, String> shapeToCodeMap = Map.ofEntries(
      Map.entry(Construct.EQUALS, "="),
      Map.entry(Construct.PLUS, "+"),
            Map.entry(Construct.MINUS, "-"),
            Map.entry(Construct.MULTIPLIER, "*"),
            Map.entry(Construct.DIVISOR, "/"),
            Map.entry(Construct.IF, "if")
    );
}
